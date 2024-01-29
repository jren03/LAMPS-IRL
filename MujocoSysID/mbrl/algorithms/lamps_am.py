# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ast import Not
from math import isnan
import os
from typing import Optional, Sequence, cast

import time
import gym
import hydra.utils
from networkx import minimum_node_cut
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.planning.sac_wrapper import SACAgent

from tqdm import tqdm

import torch.nn as nn

from mbrl.env.gym_wrappers import (
    AntMazeResetWrapper,
    GoalWrapper,
    RewardWrapper,
    TremblingHandWrapper,
    PSDPWrapper,
)
from mbrl.util.fetch_demos import fetch_demos
from mbrl.util.psdp_fetch_demos import fetch_demos as psdp_fetch_demos
from mbrl.util.am_buffers import QReplayBuffer
from mbrl.models.arch import Discriminator, DiscriminatorEnsemble
from mbrl.models.td3_bc import TD3_BC
from mbrl.util.oadam import OAdam
from stable_baselines3.common.evaluation import evaluate_policy
from mbrl.util.nn_utils import gradient_penalty
from termcolor import cprint

from mbrl.util.ema_pytorch import EMA
from pathlib import Path


def sample(env, policy, trajs, no_regret):
    # rollout trajectories using a policy and add to replay buffer
    S_curr = []
    A_curr = []
    total_trajs = 0
    s = 0
    while total_trajs < trajs:
        obs = env.reset()
        done = False
        while not done:
            S_curr.append(obs)
            act = policy.predict(obs)[0]
            A_curr.append(act)
            obs, _, done, _ = env.step(act)
            s += 1
            if done:
                total_trajs += 1
                break
    return torch.from_numpy(np.array(S_curr)), torch.from_numpy(np.array(A_curr)), s


def maybe_replace_sac_buffer(
    sac_buffer: Optional[mbrl.util.ReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> mbrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        obs, action, next_obs, reward, done = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, done)
        return new_buffer
    return sac_buffer


def psdp_rollout_in_buffer(
    model_env: mbrl.models.ModelEnv,
    initial_obs: np.ndarray,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    rollout_horizon: int,
):
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    for i in range(rollout_horizon):
        action = agent.act(obs, batched=True)
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


def nrpi_rollout_in_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):
    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


def create_env(env_name, psdp_wrapper: bool = False, f_net: Optional[nn.Module] = None):
    env = gym.make(env_name)
    if "maze" in env_name:
        env = GoalWrapper(env)
    else:
        raise NotImplementedError
    if psdp_wrapper:
        env = PSDPWrapper(env)
    if f_net is not None:
        env = RewardWrapper(env, f_net)
    return env


def eval_agent_in_model(
    model_env: mbrl.models.ModelEnv,
    test_env: gym.Env,
    f_net: nn.Module,
    agent: SACAgent,
    num_episodes: int,
    cfg: omegaconf.DictConfig,
):
    # elite_ensemble_size = model_env.dynamics_model.model.num_members
    elite_ensemble_size = model_env.dynamics_model.num_elites
    rewards = torch.zeros(num_episodes, elite_ensemble_size).to(cfg.device)
    for episode in range(num_episodes):
        initial_obs = test_env.reset().reshape(1, -1)
        model_state = model_env.reset(initial_obs_batch=initial_obs, return_as_np=True)
        model_state["obs"] = model_state["obs"].repeat((elite_ensemble_size, 1))
        done = False
        obs = initial_obs
        step = 0
        while not np.all(done) and step < cfg.overrides.epoch_length:
            action = agent.act(obs, batched=True)
            # repeat action model_env.dynamics_model.model.num_members times
            if action.shape[0] == 1:
                action = np.repeat(action, elite_ensemble_size, axis=0)
            obs, _, new_done, model_state = model_env.step(
                action, model_state, sample=False
            )
            reward = -f_net(
                torch.cat(
                    (
                        torch.from_numpy(obs),
                        torch.from_numpy(action),
                    ),
                    dim=1,
                )
                .float()
                .to(cfg.device)
            )
            # Set reward to 0 where done is True, otherwise set it to a specific value
            done = np.where(done, 1.0, new_done)
            reward = torch.where(
                torch.from_numpy(done.flatten()).bool().to(cfg.device),
                torch.tensor(0.0).to(cfg.device),
                reward,
            )
            rewards[episode] += reward
            step += 1
    rewards /= cfg.overrides.epoch_length
    rewards_mean = torch.mean(torch.mean(rewards, dim=1), dim=0).item()
    return rewards_mean, None


def evaluate(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
    cfg: omegaconf.DictConfig,
) -> float:
    avg_episode_reward = 0
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            if torch.is_tensor(reward):
                reward = reward.cpu().detach().item()
            episode_reward += reward
        avg_episode_reward += episode_reward
    avg_episode_reward /= cfg.overrides.epoch_length
    return avg_episode_reward / num_episodes


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
):
    alpha = 1
    learn_rate = cfg.disc_lr
    batch_size = 4096
    f_steps = 1
    pi_steps = 5000
    num_traj_sample = 4
    outer_steps = 100
    mean_rewards = []
    std_rewards = []
    env_steps = []
    log_interval = 5

    if cfg.reset_version == "psdp":
        cprint("Resting to PSDP", color="green", attrs=["bold"])
    elif cfg.reset_version == "nrpi":
        cprint("Resting to NRPI", color="green", attrs=["bold"])
    elif cfg.reset_version == "backward_sw":
        cprint("Resting to sliding window backwards", color="green", attrs=["bold"])
    elif cfg.reset_version == "forward_sw":
        cprint("Resting to sliding window fowards", color="green", attrs=["bold"])
    elif cfg.reset_version == "backward_range":
        cprint("Resting with range from backwards", color="green", attrs=["bold"])
    elif cfg.reset_version == "forward_range":
        cprint("Resting with range from beginning", color="green", attrs=["bold"])
    else:
        raise NotImplementedError

    env_name = cfg.overrides.env.lower().replace("gym___", "")
    if cfg.psdp_wrapper:
        (
            expert_dataset,
            expert_sa_pairs,
            qpos,
            qvel,
            goals,
            expert_reset_states,
        ) = psdp_fetch_demos(env_name, cfg)
        print(f"{expert_dataset['observations'].shape}")
    else:
        (
            expert_dataset,
            expert_sa_pairs,
            qpos,
            qvel,
            goals,
            expert_reset_states,
        ) = fetch_demos(env_name, cfg)
        print(f"{expert_dataset['observations'].shape}")
    expert_sa_pairs = expert_sa_pairs.to(cfg.device)

    env = create_env(env_name, cfg.psdp_wrapper, f_net=None)

    if cfg.train_discriminator:
        if cfg.use_ensemble:
            cprint("Using ensemble", color="green", attrs=["bold"])
            f_net = DiscriminatorEnsemble(
                env, tanh_disc=cfg.tanh_disc, clip=cfg.clip_md
            ).to(cfg.device)
        else:
            cprint("Not using ensemble", color="green", attrs=["bold"])
            f_net = Discriminator(env, tanh_disc=cfg.tanh_disc, clip=cfg.clip_md).to(
                cfg.device
            )
        cprint(
            f"Disc_lr: {learn_rate}, Disc_freq: {cfg.freq_train_discriminator}",
            color="green",
            attrs=["bold"],
        )
        f_opt = OAdam(
            f_net.parameters(),
            lr=learn_rate,
            weight_decay=cfg.overrides.model_wd if cfg.wd_md else 0,
        )
        # env = RewardWrapper(env, f_net)
    else:
        f_net = None

    # ---------------------------- Initialize Envs ------------------------------
    env = create_env(env_name, cfg.psdp_wrapper, f_net)
    test_env = create_env(env_name, cfg.psdp_wrapper, f_net=None)
    mixed_reset_env = AntMazeResetWrapper(
        GoalWrapper(gym.make(env_name)),
        qpos,
        qvel,
        goals,
        alpha=0.5,
    )
    if cfg.psdp_wrapper:
        mixed_reset_env = PSDPWrapper(mixed_reset_env)

    # env = TremblingHandWrapper(env, p_tremble=0)
    # test_env = TremblingHandWrapper(GoalWrapper(test_env), p_tremble=0)
    # if cfg.psdp_wrapper:
    #     env = PSDPWrapper(env)
    #     test_env = PSDPWrapper(test_env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    q_replay_buffer = QReplayBuffer(state_dim, action_dim)
    q_replay_buffer.add_d4rl_dataset(expert_dataset)
    pi_replay_buffer = QReplayBuffer(state_dim, action_dim)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.99,
        "tau": 0.005,
        # TD3
        "policy_noise": 0.2 * max_action,
        "noise_clip": 0.5 * max_action,
        "policy_freq": 2,
        # TD3 + BC
        "alpha": 2.5,
        "q_replay_buffer": q_replay_buffer,
        "pi_replay_buffer": pi_replay_buffer,
        "env": env,
        "f": f_net,
        "cfg": cfg,
    }
    agent = TD3_BC(**kwargs)
    if cfg.ema:
        ema_agent = EMA(agent)

    if cfg.bc_init:
        for _ in range(1):
            agent.learn(total_timesteps=int(1e4), bc=True)
    # mean_reward, std_reward = evaluate_policy(agent, test_env, n_eval_episodes=25)
    # print(100 * mean_reward)

    agent.actor.optimizer = OAdam(agent.actor.parameters())
    agent.critic.optimizer = OAdam(agent.critic.parameters())

    # ---------------------------------- LAMPS START ------------------------------------
    work_dir = work_dir or os.getcwd()
    MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
        ("true_reset_eval_mean", "TR", "float"),
        ("mixed_reset_eval_mean", "MR", "float"),
        ("real_env_eval_mean", "RE", "float"),
    ]
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    if "guest" in str(Path(Path.cwd())):
        model_dir = Path(
            f"/home/guest/dev/juntao/LAMPS-IRL/MujocoSysID/model_train_dir/{env_name}"
        )
    else:
        model_dir = Path(
            f"/share/portal/jlr429/pessimistic-irl/LAMPS-IRL/MujocoSysID/model_train_dir/{env_name}"
        )
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(
        cfg,
        obs_shape,
        act_shape,
        model_dir=model_dir if cfg.pretrained_dynamics_model else None,
    )
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    policy_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    replay_buffer.add_batch(
        expert_dataset["observations"][: cfg.algorithm.initial_exploration_steps],
        expert_dataset["actions"][: cfg.algorithm.initial_exploration_steps],
        expert_dataset["next_observations"][: cfg.algorithm.initial_exploration_steps],
        expert_dataset["rewards"][: cfg.algorithm.initial_exploration_steps],
        expert_dataset["terminals"][: cfg.algorithm.initial_exploration_steps],
    )
    expert_replay_buffer = mbrl.util.replay_buffer.ReplayBuffer(
        capacity=int(1e6),
        obs_shape=obs_shape,
        action_shape=act_shape,
        rng=rng,
    )
    expert_replay_buffer.add_batch(
        expert_dataset["observations"],
        expert_dataset["actions"],
        expert_dataset["next_observations"],
        expert_dataset["rewards"],
        expert_dataset["terminals"],
    )
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, None, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
        scheduler_config=cfg.decay_lr_scheduler if cfg.decay_lr else None,
    )

    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        agent,
        {},
        replay_buffer=replay_buffer,
        additional_buffer=policy_buffer,
    )
    sac_buffer = None
    # ---------------------------------- LAMPS END ------------------------------------

    # ---------------------------------- ORIGINAL FILTER LOOP --------------------------
    env_steps = 0
    disc_steps = 0
    print(f"Training {env_name}, saving to {work_dir}")
    tbar = tqdm(range(cfg.overrides.num_steps), ncols=0, mininterval=10)
    epoch = 0
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )
        agent.pi_replay_buffer = sac_buffer
        obs, done = None, False
        # for step in range(pi_steps):
        # for epoch in range(cfg.overrides.epoch_length):
        steps_epoch = 0
        while steps_epoch < cfg.overrides.epoch_length:
            if steps_epoch == 0 or done:
                obs, done = env.reset(), False
            try:
                next_obs, _, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                    env,
                    obs,
                    agent,
                    {},
                    replay_buffer=agent.pi_replay_buffer,
                    additional_buffer=replay_buffer,
                )
            except Exception as e:
                print(f"Lost Connection: {e}")
                # possibly due to nan params in actor
                breakpoint()
                # try:
                #     env.close()
                # except:
                #     env = None
                # env = create_env(env_name, cfg.psdp_wrapper, f_net)
                # obs, done = env.reset(), False
                # steps_epoch = 0
                # continue

            (
                exp_obs,
                exp_next_obs,
                exp_act,
                exp_reward,
                exp_done,
            ) = expert_replay_buffer.sample_one()
            replay_buffer.add(exp_obs, exp_act, exp_next_obs, exp_reward, exp_done)

            if (env_steps + 1) % int(cfg.overrides.freq_train_model / 2) == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )

                if cfg.reset_version == "nrpi":
                    # always reset to some expert state
                    nrpi_rollout_in_buffer(
                        model_env,
                        expert_replay_buffer,
                        agent,
                        sac_buffer,
                        cfg.algorithm.sac_samples_action,
                        rollout_length,
                        rollout_batch_size,
                    )
                else:
                    row_indices = np.random.choice(
                        len(expert_reset_states), size=rollout_batch_size, replace=True
                    )
                    if cfg.reset_version == "psdp":
                        # reset to i*T/N
                        col = int(
                            env_steps
                            / cfg.overrides.num_steps
                            * cfg.overrides.epoch_length
                        )
                        indices = np.stack(
                            (row_indices, col * np.ones_like(row_indices)), axis=-1
                        )
                    elif cfg.reset_version == "backward_sw":
                        # reset to [-i/n * T, (i/n + 0.05) * T]
                        percent_remaining = 1 - env_steps / cfg.overrides.num_steps
                        low = int(percent_remaining * cfg.overrides.epoch_length)
                        high = min(
                            cfg.overrides.epoch_length,
                            int((percent_remaining + 0.1) * cfg.overrides.epoch_length),
                        )
                        column_indices = np.random.randint(
                            low, high, size=len(row_indices)
                        )
                        indices = np.stack((row_indices, column_indices), axis=-1)
                    elif cfg.reset_version == "backward_range":
                        # reset to [-i/N * T, N]
                        low = int(
                            cfg.overrides.epoch_length
                            * (1 - env_steps / cfg.overrides.num_steps)
                        )
                        high = cfg.overrides.epoch_length
                        column_indices = np.random.randint(
                            low, high, size=len(row_indices)
                        )
                        indices = np.stack((row_indices, column_indices), axis=-1)
                    elif cfg.reset_version == "forward_sw":
                        # reset to [i/n * T, (i/n + 0.1) * T]
                        percent_completed = env_steps / cfg.overrides.num_steps
                        low = int(percent_completed * cfg.overrides.epoch_length)
                        high = int(
                            (percent_completed + 0.1) * cfg.overrides.epoch_length
                        )
                        high = min(cfg.overrides.epoch_length, high)
                        column_indices = np.random.randint(
                            low, high, size=len(row_indices)
                        )
                        indices = np.stack((row_indices, column_indices), axis=-1)
                    elif cfg.reset_version == "forward_range":
                        # reset to [0, i/N * T]
                        low = 0
                        high = max(
                            int(
                                cfg.overrides.epoch_length
                                * (env_steps / cfg.overrides.num_steps)
                            ),
                            100,
                        )
                        column_indices = np.random.randint(
                            low, high, size=len(row_indices)
                        )
                        indices = np.stack((row_indices, column_indices), axis=-1)
                    reset_states = expert_reset_states[indices[:, 0], indices[:, 1]]
                    psdp_rollout_in_buffer(
                        model_env,
                        reset_states,
                        agent,
                        sac_buffer,
                        rollout_length,
                    )

            for _ in range(cfg.overrides.num_sac_updates_per_step):
                agent.step(bc=False)
                updates_made += 1
                if cfg.ema:
                    ema_agent.update()
                for param in agent.actor.parameters():
                    if torch.isnan(param).any():
                        print("Actor has nan params")
                        breakpoint()

            if (
                cfg.train_discriminator
                and updates_made % cfg.freq_train_discriminator == 0
            ):
                # agent.learn(total_timesteps=pi_steps, log_interval=1000)
                # steps += pi_steps
                # print(f"Training Discriminator at step {env_steps}")
                if not disc_steps == 0:
                    learning_rate_used = learn_rate / disc_steps
                else:
                    learning_rate_used = learn_rate
                f_opt = OAdam(
                    f_net.parameters(),
                    lr=learning_rate_used,
                    weight_decay=cfg.overrides.model_wd if cfg.wd_md else 0,
                )

                S_curr, A_curr, s = sample(
                    test_env, agent, num_traj_sample, no_regret=False
                )
                learner_sa_pairs = torch.cat((S_curr, A_curr), dim=1).to(cfg.device)

                for _ in range(f_steps):
                    learner_sa = learner_sa_pairs[
                        np.random.choice(len(learner_sa_pairs), batch_size)
                    ]
                    expert_sa = expert_sa_pairs[
                        np.random.choice(len(expert_sa_pairs), batch_size)
                    ]
                    f_opt.zero_grad()
                    f_learner = f_net(learner_sa.float())
                    f_expert = f_net(expert_sa.float())
                    gp = gradient_penalty(learner_sa, expert_sa, f_net)
                    loss = f_expert.mean() - f_learner.mean() + 10 * gp
                    loss.backward()
                    f_opt.step()
                disc_steps += 1

            if env_steps % cfg.freq_eval == 0 and env_steps != 0:
                # breakpoint()
                if cfg.ema:
                    eval_agent = ema_agent
                else:
                    eval_agent = agent
                mean_reward, _ = evaluate_policy(
                    eval_agent, test_env, n_eval_episodes=15
                )
                mean_reward = mean_reward * 100
                # try:
                real_env_eval_mean = evaluate(env, eval_agent, num_episodes=15, cfg=cfg)
                try:
                    true_reset_eval_mean, _ = eval_agent_in_model(
                        model_env, test_env, f_net, eval_agent, 5, cfg
                    )
                    mixed_reset_eval_mean, _ = eval_agent_in_model(
                        model_env, mixed_reset_env, f_net, eval_agent, 5, cfg
                    )
                except Exception as e:
                    print(e)
                    breakpoint()

                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "env_step": env_steps,
                        "episode_reward": mean_reward,
                        "true_reset_eval_mean": true_reset_eval_mean,
                        "mixed_reset_eval_mean": mixed_reset_eval_mean,
                        "real_env_eval_mean": real_env_eval_mean,
                    },
                )
                # except:
                #     logger.log_data(
                #         mbrl.constants.RESULTS_LOG_NAME,
                #         {
                #             "env_step": env_steps,
                #             "episode_reward": mean_reward,
                #             "true_reset_eval_mean": 0.0,
                #             "mixed_reset_eval_mean": 0.0,
                #             "real_env_eval_mean": 0.0,
                #         },
                #     )
                #     print("{0} Iteration: {1}".format(int(env_steps), mean_reward))

            tbar.update(1)
            env_steps += 1
            steps_epoch += 1
            obs = next_obs

        epoch += 1
    # ----------------------------- ORIGINAL FILTER LOOP END --------------------------
