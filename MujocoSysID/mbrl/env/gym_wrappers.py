import sys
import os
import gym
import torch
import numpy as np
from mbrl.util.common import PrintColors, HiddenPrints
from termcolor import cprint


def _get_env_name(env):
    env_str = str(env)
    env_str = env_str.rsplit("<", 1)[1].split()[0]
    if ">" in env_str:
        env_str = env_str.split(">", 1)[0]
    return env_str


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, function):
        super().__init__(env)
        self.env = env
        self.cur_state = None
        self.function = function
        self.low = env.action_space.low
        self.high = env.action_space.high

    def reset(self):
        obs = self.env.reset()
        self.cur_state = obs
        return obs

    def step(self, action):
        next_state, _, done, info = self.env.step(action)
        # combine action and state
        sa_pair = np.concatenate((self.cur_state, action))
        reward = -(
            self.function.forward(torch.tensor(sa_pair, dtype=torch.float).to("cuda"))
        )
        self.cur_state = next_state

        return next_state, reward, done, info


class TremblingHandWrapper(gym.Wrapper):
    def __init__(self, env, p_tremble=0.01):
        super().__init__(env)
        self.env = env
        self.p_tremble = p_tremble
        cprint(f"{self.p_tremble=}", attrs=["bold"])

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if np.random.uniform() < self.p_tremble:
            action = self.env.action_space.sample()
        return self.env.step(action)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class GoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,))

    def reset(self):
        with HiddenPrints():
            obs = self.env.reset()
            goal = self.env.target_goal
            return np.concatenate([obs, goal])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        goal = self.env.target_goal
        return np.concatenate([obs, goal]), rew, done, info


class AntMazeResetWrapper(gym.Wrapper):
    def __init__(self, env, qpos, qvel, G, alpha=1):
        super().__init__(env)
        self.env = env
        self.alpha = alpha
        self.qpos = qpos
        self.qvel = qvel
        self.G = G
        self.t = 0
        self.T = 700
        self.rng = np.random.default_rng()
        cprint(f"AntMazeResetWrapper: {self.alpha=}", attrs=["bold"])

    def reset(self):
        obs = self.env.reset()
        if self.rng.random() < self.alpha:
            idx = np.random.choice(len(self.qpos))
            t = np.random.choice(len(self.qpos[idx]))
            with HiddenPrints():
                self.env.set_target(tuple(self.G[idx][t]))
            self.env.set_state(self.qpos[idx][t], self.qvel[idx][t])
            self.t = t
            obs = self.env.env.wrapped_env._get_obs()
            goal = self.env.target_goal
            obs = np.concatenate([obs, goal])
        else:
            self.t = 0
        return obs

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        self.t += 1
        if self.t >= self.T:
            done = True
        return next_obs, rew, done, info
