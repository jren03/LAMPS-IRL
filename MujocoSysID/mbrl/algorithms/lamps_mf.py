# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Sequence, cast
import time

import gym
import hydra.utils
import numpy as np
import omegaconf
import torch
import pprint

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.env.gym_wrappers import ResetWrapper
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder
from mbrl.util.fetch_demos import fetch_demos
from mbrl.models.discriminator import Discriminator, DiscriminatorEnsemble
from mbrl.util.oadam import OAdam
from mbrl.util.common import gradient_penalty, PrintColors
from mbrl.util.discriminator_replay_buffer import DiscriminatorReplayBuffer
from ema_pytorch import EMA

import d4rl
from tqdm import tqdm
from ema_pytorch import EMA

import stable_baselines3 as sb3
from pathlib import Path
import warnings


MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
    ("sac_reset_ratio", "SRR", "float"),
    ("disc_loss", "DL", "float"),
]


def rollout_model_and_populate_sac_buffer(
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


def evaluate(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
    video_recorder: VideoRecorder,
    maze=False,
) -> float:
    avg_episode_reward = 0
    success = 0
    for episode in range(num_episodes):
        obs = env.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        if maze:
            success += episode_reward > 0
            avg_episode_reward += episode_reward
        else:
            avg_episode_reward += episode_reward
    if maze:
        return avg_episode_reward / num_episodes, success / num_episodes
    return avg_episode_reward / num_episodes


def sample(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
    replay_buffer: DiscriminatorReplayBuffer,
    no_regret=False,
) -> float:
    states, actions = [], []
    env_steps = 0
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            states.append(obs)
            action = agent.act(obs)
            actions.append(action)
            obs, _, done, _ = env.step(action)
            env_steps += 1
    states_np, actions_np = np.array(states), np.array(actions)
    if no_regret:
        replay_buffer.add(states_np, actions_np)
    return states_np, actions_np, env_steps


def sample_from_learned_model(
    env: gym.Env,
    model_env: mbrl.models.ModelEnv,
    agent: sb3.SAC,
    num_episodes: int,
    rollout_horizon: int,
):
    states, actions = [], []
    env_steps = 0
    for episode in range(num_episodes):
        real_env_obs = env.reset()
        real_env_obs = real_env_obs.reshape(1, -1)
        model_state = model_env.reset(
            initial_obs_batch=cast(np.ndarray, real_env_obs),
            return_as_np=True,
        )
        obs = real_env_obs
        for _ in range(rollout_horizon):
            breakpoint()
            action = agent.predict(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]
            states.append(obs)
            actions.append(action)
            obs, _, done, model_state = model_env.step(
                action, model_state, sample=False
            )
            env_steps += 1
            if done:
                break
    states_np, actions_np = np.array(states), np.array(actions)
    return states_np, actions_np, env_steps


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


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)
    pp = pprint.PrettyPrinter(indent=4)

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    is_maze = "maze" in cfg.overrides.env
    expert_dataset, qpos, qvel = fetch_demos(
        cfg.overrides.env,
        zero_out_rewards=cfg.train_discriminator,
        use_mbrl_demos=cfg.use_mbrl_demos,
    )
    env = ResetWrapper(env, qpos=qpos, qvel=qvel)

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32

    # ------------ Fill expert buffer ---------------------
    expert_replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    expert_replay_buffer.add_batch(
        expert_dataset["observations"][: cfg.overrides.expert_size],
        expert_dataset["actions"][: cfg.overrides.expert_size],
        expert_dataset["next_observations"][: cfg.overrides.expert_size],
        expert_dataset["rewards"][: cfg.overrides.expert_size],
        expert_dataset["terminals"][: cfg.overrides.expert_size],
    )

    # --------------- SAC reset ratio schedule -----------------
    model_reset_schedule = np.array(
        [
            [
                cfg.model_schedule.start_ratio,
                cfg.model_schedule.mid_ratio,
                cfg.model_schedule.m1,
            ],
            [
                cfg.model_schedule.mid_ratio,
                cfg.model_schedule.end_ratio,
                cfg.model_schedule.m2,
            ],
        ]
    )
    model_reset_ratio = model_reset_schedule[0, 0]
    model_ratio_lag = 0
    sac_reset_schedule = np.array(
        [
            [
                cfg.sac_schedule.start_ratio,
                cfg.sac_schedule.mid_ratio,
                cfg.sac_schedule.m1,
            ],
            [
                cfg.sac_schedule.mid_ratio,
                cfg.sac_schedule.end_ratio,
                cfg.sac_schedule.m2,
            ],
        ]
    )
    sac_reset_ratio = sac_reset_schedule[0, 0]
    sac_ratio_lag = 0

    # -------------- discriminator lr schedule ------------------
    disc_lr = cfg.disc.lr
    if cfg.no_regret:
        print(f"{PrintColors.OKBLUE}No regret discriminator training")
        print(PrintColors.ENDC)
    else:
        print(f"{PrintColors.OKBLUE}Best response discriminator training")
        print(PrintColors.ENDC)
    drb = DiscriminatorReplayBuffer(obs_shape[0], act_shape[0])

    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    if cfg.train_discriminator:
        if cfg.disc_ensemble:
            print(
                f"{PrintColors.OKBLUE}Training with discriminator function ENSEMBLE{PrintColors.ENDC}"
            )
            f_net = DiscriminatorEnsemble(
                env, n_discriminators=cfg.n_discs, reduction=cfg.disc_ensemble_reduction
            ).to(cfg.device)
        else:
            print(
                f"{PrintColors.OKBLUE}Training with discriminator function REGULAR{PrintColors.ENDC}"
            )
            f_net = Discriminator(env).to(cfg.device)
        f_opt = OAdam(f_net.parameters(), lr=disc_lr)
        if cfg.disc.ema:
            print(PrintColors.OKBLUE + "Using EMA for discriminator" + PrintColors.ENDC)
            ema = EMA(
                f_net,
                update_after_step=20,
                update_every=2,
                inv_gamma=1,
                power=3 / 4,
            )
        agent.sac_agent.add_f_net(f_net)
    else:
        print(
            f"{PrintColors.OKBLUE}Training with ground truth rewards{PrintColors.ENDC}"
        )

    best_eval_reward = -np.inf
    epoch = 0
    disc_steps = 0
    sac_buffer = None

    agent.sac_agent.reset_optimizers()
    tbar = tqdm(range(cfg.overrides.num_steps), ncols=0)
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
        obs, done = None, False

        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or done:
                obs, done = env.reset(), False
            # --- Doing env step and adding to model dataset ---
            # start_time = time.time()
            obs, _, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, sac_buffer
            )

            # --------------- Agent Training -----------------

            # start_time = time.time()
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                if (env_steps) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    sac_buffer
                ) < cfg.overrides.sac_batch_size:
                    break  # only update every once in a while

                agent.sac_agent.bc_reg_update_parameters(
                    sac_buffer,
                    expert_replay_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    logger,
                    reverse_mask=True,
                )
                agent.sac_agent.updates_made += 1
                agent.sac_agent.step_lr()
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)
            # print(f"Time for agent training: {time.time() - start_time}")

            # ------ Discriminator Training ------
            if (
                cfg.train_discriminator
                and not cfg.update_with_model
                and agent.sac_agent.updates_made != 0
                and (agent.sac_agent.updates_made) % cfg.disc.freq_train_disc == 0
            ):
                print("HERE")
                if not disc_steps == 0:
                    disc_lr = cfg.disc.lr / disc_steps
                else:
                    disc_lr = cfg.disc.lr
                f_opt = OAdam(f_net.parameters(), lr=disc_lr)

                S_curr, A_curr, s = sample(
                    test_env,
                    agent,
                    cfg.disc.num_traj_samples,
                    drb,
                    cfg.no_regret,
                )
                learner_sa_pairs = torch.cat(
                    (torch.from_numpy(S_curr), torch.from_numpy(A_curr)), dim=1
                ).to(cfg.device)
                # env_steps += s    # * ignore env_steps for discriminator training
                # tbar.update(s)
                for _ in range(cfg.disc.num_updates_per_step):
                    learner_sa = learner_sa_pairs[
                        np.random.choice(len(learner_sa_pairs), cfg.disc.batch_size)
                    ]
                    expert_batch = expert_replay_buffer.sample(cfg.disc.batch_size)
                    expert_s, expert_a, *_ = cast(
                        mbrl.types.TransitionBatch, expert_batch
                    ).astuple()
                    expert_sa = torch.cat(
                        (torch.from_numpy(expert_s), torch.from_numpy(expert_a)),
                        dim=1,
                    ).to(cfg.device)
                    f_opt.zero_grad()
                    f_learner = f_net(learner_sa.float())
                    f_expert = f_net(expert_sa.float())
                    gp = gradient_penalty(learner_sa, expert_sa, f_net)
                    disc_loss = f_expert.mean() - f_learner.mean() + 10 * gp
                    disc_loss.backward()
                    f_opt.step()
                    if cfg.disc.ema:
                        ema.update()
                disc_steps += 1
                agent.sac_agent.reset_optimizers()
                # print(f"REEE 2: {updates_made}")

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                epoch += 1
            if (env_steps + 1) % cfg.eval_frequency == 0:
                if not is_maze:
                    # start_time = time.time()
                    avg_reward = evaluate(
                        test_env,
                        agent,
                        cfg.algorithm.num_eval_episodes,
                        video_recorder,
                        is_maze,
                    )
                    logger.log_data(
                        mbrl.constants.RESULTS_LOG_NAME,
                        {
                            "epoch": epoch,
                            "env_step": env_steps,
                            "episode_reward": avg_reward,
                            "rollout_length": rollout_length,
                            "sac_reset_ratio": sac_reset_ratio,
                            "disc_loss": disc_loss,
                        },
                    )
                    # print(f"Time for evaluation: {time.time() - start_time}")
                else:
                    avg_reward, success_rate = evaluate(
                        test_env,
                        agent,
                        cfg.algorithm.num_eval_episodes,
                        video_recorder,
                        is_maze,
                    )
                    logger.log_data(
                        mbrl.constants.RESULTS_LOG_NAME,
                        {
                            "epoch": epoch,
                            "env_step": env_steps,
                            "episode_reward": avg_reward,
                            "success_rate": success_rate,
                            "rollout_length": rollout_length,
                        },
                    )
            tbar.update(1)
            env_steps += 1
            obs = next_obs

    return np.float32(best_eval_reward)
