# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ast import Not
import os
from typing import Optional, Sequence, cast

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
from mbrl.third_party.pytorch_sac import VideoRecorder

from tqdm import tqdm

from mbrl.util.fetch_demos import fetch_demos

import d4rl

from mbrl.env.gym_wrappers import (
    AntMazeResetWrapper,
    GoalWrapper,
    RewardWrapper,
    TremblingHandWrapper,
)
from mbrl.util.fetch_demos import fetch_demos
from mbrl.util.am_buffers import QReplayBuffer
from mbrl.models.arch import Discriminator
from mbrl.models.td3_bc import TD3_BC
from mbrl.util.oadam import OAdam
from stable_baselines3.common.evaluation import evaluate_policy
from mbrl.util.nn_utils import gradient_penalty


def sample(env, policy, trajs, no_regret):
    # rollout trajectories using a policy and add to replay buffer
    S_curr = []
    A_curr = []
    total_trajs = 0
    alpha = env.alpha
    env.alpha = 0
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
    env.alpha = alpha
    return torch.from_numpy(np.array(S_curr)), torch.from_numpy(np.array(A_curr)), s


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
):
    alpha = 1
    learn_rate = 8e-3
    batch_size = 4096
    f_steps = 1
    pi_steps = 5000
    num_traj_sample = 4
    outer_steps = 100
    mean_rewards = []
    std_rewards = []
    env_steps = []
    log_interval = 5

    env_name = cfg.overrides.env.lower().replace("gym___", "")
    expert_dataset, expert_sa_pairs, qpos, qvel, goals = fetch_demos(env_name)
    expert_sa_pairs = expert_sa_pairs.to(cfg.device)

    if "maze" in env_name:
        env = AntMazeResetWrapper(GoalWrapper(env), qpos, qvel, goals)
    else:
        raise NotImplementedError

    env.alpha = alpha
    f_net = Discriminator(env).to(cfg.device)
    f_opt = OAdam(f_net.parameters(), lr=learn_rate)
    env = RewardWrapper(env, f_net)

    env = TremblingHandWrapper(env, p_tremble=0)
    test_env = TremblingHandWrapper(GoalWrapper(test_env), p_tremble=0)

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
    }
    agent = TD3_BC(**kwargs)
    for _ in range(1):
        agent.learn(total_timesteps=int(1e4), bc=True)
        mean_reward, std_reward = evaluate_policy(agent, test_env, n_eval_episodes=25)
        print(100 * mean_reward)

    agent.actor.optimizer = OAdam(agent.actor.parameters())
    agent.critic.optimizer = OAdam(agent.critic.parameters())

    # ---------------------------------- LAMPS START ------------------------------------
    work_dir = work_dir or os.getcwd()
    MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT
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
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
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
    print(f"Training {env_name}")
    tbar = tqdm(range(500_000), ncols=0, mininterval=10)
    while env_steps < 500_000:
        obs, done = None, False
        # for step in range(pi_steps):
        for epoch in range(700):
            if epoch == 0 or done:
                obs, done = env.reset(), False
            next_obs, _, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env,
                obs,
                agent,
                {},
                replay_buffer=agent.pi_replay_buffer,
                additional_buffer=replay_buffer,
            )
            agent.step(bc=False)
            tbar.update(1)
            env_steps += 1
            obs = next_obs

            if env_steps % cfg.freq_train_discriminator == 0:
                # agent.learn(total_timesteps=pi_steps, log_interval=1000)
                # steps += pi_steps
                print(f"Training Discriminator at step {env_steps}")
                if not disc_steps == 0:
                    learning_rate_used = learn_rate / disc_steps
                else:
                    learning_rate_used = learn_rate
                f_opt = OAdam(f_net.parameters(), lr=learning_rate_used)

                S_curr, A_curr, s = sample(env, agent, num_traj_sample, no_regret=False)
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

            if env_steps % cfg.freq_eval == 0:
                mean_reward, std_reward = evaluate_policy(
                    agent, test_env, n_eval_episodes=25
                )
                mean_reward = mean_reward * 100
                std_reward = std_reward * 100
                mean_rewards.append(mean_reward)
                std_rewards.append(std_reward)
                # env_steps.append(steps)
                print("{0} Iteration: {1}".format(int(env_steps), mean_reward))
    # ----------------------------- ORIGINAL FILTER LOOP END --------------------------
