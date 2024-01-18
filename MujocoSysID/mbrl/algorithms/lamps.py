# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from tabnanny import verbose
from typing import Optional, Sequence, cast, Dict, Tuple
import time

import gym
import hydra.utils
import numpy as np
import omegaconf
import torch
import pprint

from stable_baselines3.common.evaluation import evaluate_policy

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.env.gym_wrappers import ResetWrapper, RewardWrapper, TremblingHandWrapper
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder
from mbrl.util.fetch_demos import fetch_demos
from mbrl.models.discriminator import Discriminator, DiscriminatorEnsemble
from mbrl.util.oadam import OAdam
from mbrl.util.common import gradient_penalty, PrintColors
from mbrl.util.discriminator_replay_buffer import DiscriminatorReplayBuffer
from mbrl.util.hybrid_replay_buffer import HybridReplayBuffer
from mbrl.util.sac_relabel_rewards import SACRelabelRewards
from typing import Callable, Union

import d4rl
from tqdm import tqdm

import stable_baselines3 as sb3
from pathlib import Path
import warnings


MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
    ("sac_reset_ratio", "SRR", "float"),
]


def step_env_and_add_to_buffer(
    env: gym.Env,
    obs: np.ndarray,
    agent: SACRelabelRewards,
    replay_buffer: mbrl.util.replay_buffer.ReplayBuffer,
    additional_buffer: mbrl.util.replay_buffer.ReplayBuffer,
) -> Tuple[np.ndarray, float, bool, Dict]:
    # according to collect_rollouts in https://github.com/DLR-RM/stable-baselines3/blob/8e5ede783f23070acd2f209b8b39ccefd6840fa4/stable_baselines3/common/off_policy_algorithm.py#L512
    action = agent.predict(obs, deterministic=False)[0]
    next_obs, reward, done, info = env.step(action)
    replay_buffer.add(obs, action, next_obs, reward, done)
    additional_buffer.add(obs, action, next_obs, reward, done)
    return next_obs, reward, done, info


def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACRelabelRewards,
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
        action = agent.predict(obs, deterministic=False)[0]
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        agent.replay_buffer.add_batch(
            obs[~accum_dones],
            pred_next_obs[~accum_dones],
            action[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


def sample(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
) -> float:
    states, actions = [], []
    env_steps = 0
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            states.append(obs)
            action = agent.predict(obs)[0]
            actions.append(action)
            obs, _, done, _ = env.step(action)
            env_steps += 1
    states_np, actions_np = np.array(states), np.array(actions)
    return states_np, actions_np, env_steps


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
    p_tremble: int = 0.0,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    is_maze = "maze" in cfg.overrides.env
    expert_dataset, qpos, qvel = fetch_demos(
        cfg.overrides.env,
        zero_out_rewards=cfg.train_discriminator,
        use_mbrl_demos=cfg.use_mbrl_demos,
    )

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    model_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    learner_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env),
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=model_buffer,
        additional_buffer=learner_buffer,
    )

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
    model_buffer.add_batch(
        expert_dataset["observations"][: cfg.algorithm.initial_exploration_steps],
        expert_dataset["actions"][: cfg.algorithm.initial_exploration_steps],
        expert_dataset["next_observations"][: cfg.algorithm.initial_exploration_steps],
        expert_dataset["rewards"][: cfg.algorithm.initial_exploration_steps],
        expert_dataset["terminals"][: cfg.algorithm.initial_exploration_steps],
    )
    expert_replay_buffer.add_batch(
        expert_dataset["observations"][-cfg.overrides.expert_size :],
        expert_dataset["actions"][-cfg.overrides.expert_size :],
        expert_dataset["next_observations"][-cfg.overrides.expert_size :],
        expert_dataset["rewards"][-cfg.overrides.expert_size :],
        expert_dataset["terminals"][-cfg.overrides.expert_size :],
    )

    disc_lr = cfg.disc.start_lr
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
                f"{PrintColors.OKBLUE}Training with discriminator function NO ENSEMBLE{PrintColors.ENDC}"
            )
            f_net = Discriminator(env).to(cfg.device)
        f_opt = OAdam(f_net.parameters(), lr=disc_lr)
        model_env = mbrl.models.ModelEnv(
            env, dynamics_model, termination_fn, f_net, generator=torch_generator
        )
    else:
        print(
            f"{PrintColors.OKBLUE}Training with ground truth rewards{PrintColors.ENDC}"
        )
        model_env = mbrl.models.ModelEnv(
            env, dynamics_model, termination_fn, None, generator=torch_generator
        )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )

    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    total_sac_buffer_size = int(
        cfg.overrides.rollout_schedule[-1] * rollout_batch_size * trains_per_epoch
    )

    if not cfg.hyirl:
        env = ResetWrapper(env, qpos=qpos, qvel=qvel)
    env = RewardWrapper(env, f_net)
    env = TremblingHandWrapper(env, p_tremble=p_tremble)
    test_env = TremblingHandWrapper(test_env, p_tremble=p_tremble)
    agent = SACRelabelRewards(
        f_net,
        "MlpPolicy",
        env,
        verbose=0,
        policy_kwargs=dict(net_arch=[256, 256]),
        ent_coef="auto",
        learning_rate=linear_schedule(7.3e-4),
        train_freq=64,
        gradient_steps=64,
        gamma=0.98,
        tau=0.02,
        buffer_size=total_sac_buffer_size,
        device=cfg.device,
    )
    agent.actor.optimizer = OAdam(agent.actor.parameters())
    agent.critic.optimizer = OAdam(agent.critic.parameters())
    agent.replay_buffer = HybridReplayBuffer(
        agent.buffer_size,
        agent.observation_space,
        agent.action_space,
        agent.device,
        1,
        agent.optimize_memory_usage,
        expert_data=expert_dataset,
        balanced_sampling=cfg.hyirl,
        fixed_hybrid_schedule=False,
    )
    epoch = 0
    disc_steps = 0
    env_steps = 0
    tbar = tqdm(range(cfg.overrides.num_steps), ncols=0)
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        agent.replay_buffer.maybe_update_buffer_capacity(sac_buffer_capacity)

        obs, done = None, False

        # expand out the learn() from sb3 to allow for rollouts in the model
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or done:
                obs = env.reset()
                done = False
            obs, _, done, _ = step_env_and_add_to_buffer(
                env, obs, agent, model_buffer, additional_buffer=learner_buffer
            )
            (
                exp_obs,
                exp_next_obs,
                exp_act,
                exp_reward,
                exp_done,
            ) = expert_replay_buffer.sample_one()
            model_buffer.add(exp_obs, exp_act, exp_next_obs, exp_reward, exp_done)

            env_steps += 1
            tbar.update(1)

            # -------------------- Model Training --------------------
            if (env_steps) % int(cfg.overrides.freq_train_model / 2) == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    model_buffer,
                    work_dir=work_dir,
                )

                reset_sac = rng.random() < cfg.sac_reset_ratio
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    expert_replay_buffer if reset_sac else learner_buffer,
                    agent,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

            # -------------------- SAC Training --------------------
            if (
                agent.replay_buffer.size() > agent.batch_size
                and (env_steps) % agent.train_freq == 0
            ):
                # train agent using hybrid replay buffer
                agent.train(
                    gradient_steps=agent.gradient_steps, batch_size=agent.batch_size
                )
        epoch += 1

        # ------ Discriminator Training ------
        if (
            cfg.train_discriminator
            and not cfg.train_disc_in_model
            and env_steps % cfg.disc.freq_train_disc == 0
        ):
            if not disc_steps == 0:
                disc_lr = cfg.disc.start_lr / disc_steps
            else:
                disc_lr = cfg.disc.start_lr
            f_opt = OAdam(f_net.parameters(), lr=disc_lr)
            S_curr, A_curr, s = sample(
                test_env,
                agent,
                cfg.disc.num_traj_samples,
            )
            learner_sa_pairs = torch.cat(
                (torch.from_numpy(S_curr), torch.from_numpy(A_curr)), dim=1
            ).to(cfg.device)
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
                loss = f_expert.mean() - f_learner.mean() + 10 * gp
                loss.backward()
                f_opt.step()
            disc_steps += 1

        # ------ Epoch ended (evaluate and save model) ------
        if (env_steps) % cfg.eval_frequency == 0:
            mean_reward, std_reward = evaluate_policy(
                agent, test_env, n_eval_episodes=10
            )
            logger.log_data(
                mbrl.constants.RESULTS_LOG_NAME,
                {
                    "epoch": epoch,
                    "env_step": env_steps,
                    "episode_reward": mean_reward,
                    "sac_reset_ratio": agent.replay_buffer.ratio,
                },
            )

    print("SUCCESS")
