# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ast import Not
import os
from typing import Optional, Sequence, cast

import gym
import hydra.utils
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
    cur_env = gym.make(env_name)

    expert_dataset, expert_sa_pairs, qpos, qvel, goals = fetch_demos(env_name)
    expert_sa_pairs = expert_sa_pairs.to(cfg.device)

    if "maze" in env_name:
        cur_env = AntMazeResetWrapper(GoalWrapper(cur_env), qpos, qvel, goals)
    else:
        raise NotImplementedError

    cur_env.alpha = alpha
    f_net = Discriminator(cur_env).to(cfg.device)
    f_opt = OAdam(f_net.parameters(), lr=learn_rate)
    cur_env = RewardWrapper(cur_env, f_net)

    cur_env = TremblingHandWrapper(cur_env, p_tremble=0)
    eval_env = TremblingHandWrapper(GoalWrapper(gym.make(env_name)), p_tremble=0)

    state_dim = cur_env.observation_space.shape[0]
    action_dim = cur_env.action_space.shape[0]
    max_action = float(cur_env.action_space.high[0])

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
        "env": cur_env,
        "f": f_net,
    }
    pi = TD3_BC(**kwargs)
    for _ in range(1):
        pi.learn(total_timesteps=int(1e4), bc=True)
        mean_reward, std_reward = evaluate_policy(pi, eval_env, n_eval_episodes=25)
        print(100 * mean_reward)

    pi.actor.optimizer = OAdam(pi.actor.parameters())
    pi.critic.optimizer = OAdam(pi.critic.parameters())

    steps = 0
    print(f"Training {env_name}")
    for outer in range(outer_steps):
        if not outer == 0:
            learning_rate_used = learn_rate / outer
        else:
            learning_rate_used = learn_rate
        f_opt = OAdam(f_net.parameters(), lr=learning_rate_used)

        pi.learn(total_timesteps=pi_steps, log_interval=1000)
        steps += pi_steps

        S_curr, A_curr, s = sample(cur_env, pi, num_traj_sample, no_regret=False)
        steps += s
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

        if outer % log_interval == 0:
            mean_reward, std_reward = evaluate_policy(pi, eval_env, n_eval_episodes=25)
            mean_reward = mean_reward * 100
            std_reward = std_reward * 100
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
            env_steps.append(steps)
            print("{0} Iteration: {1}".format(int(outer), mean_reward))
