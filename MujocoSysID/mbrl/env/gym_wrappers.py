import gym
import numpy as np
from mbrl.util.common import PrintColors as PC
from mbrl.util.common import HiddenPrints


def _get_env_name(env):
    env_str = str(env)
    env_str = env_str.rsplit("<", 1)[1].split()[0]
    if ">" in env_str:
        env_str = env_str.split(">", 1)[0]
    return env_str


class TremblingHandWrapper(gym.Wrapper):
    def __init__(self, env, p_tremble=0.01):
        super().__init__(env)
        self.env = env
        self.p_tremble = p_tremble
        print(f"{PC.BOLD}{_get_env_name(self.env)} {p_tremble=}{PC.ENDC}")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if np.random.uniform() < self.p_tremble:
            action = self.env.action_space.sample()
        return self.env.step(action)


class LearnerRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.low = env.action_space.low
        self.high = env.action_space.high
        print(PC.BOLD + "Wrapping env in LearnerRewardWrapper" + PC.ENDC)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        next_state, _, done, info = self.env.step(action)
        reward = 0.0
        return next_state, reward, done, info


class GoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,))
        print(f"{PC.OKBLUE}Goal Wrapping{_get_env_name(self.env)}{PC.ENDC}")

    def reset(self):
        with HiddenPrints():
            obs = self.env.reset()
            goal = self.env.target_goal
            return np.concatenate([obs, goal])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        goal = self.env.target_goal
        return np.concatenate([obs, goal]), rew, done, info
