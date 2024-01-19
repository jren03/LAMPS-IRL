import gym
import numpy as np
from mbrl.util.common import PrintColors, HiddenPrints


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
        print(
            f"{PrintColors.BOLD}{_get_env_name(self.env)} {p_tremble=}{PrintColors.ENDC}"
        )

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if np.random.uniform() < self.p_tremble:
            action = self.env.action_space.sample()
        return self.env.step(action)


class GoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,))
        print(
            f"{PrintColors.OKBLUE}Goal Wrapping{_get_env_name(self.env)}{PrintColors.ENDC}"
        )

    def reset(self):
        with HiddenPrints():
            obs = self.env.reset()
            goal = self.env.target_goal
            return np.concatenate([obs, goal])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        goal = self.env.target_goal
        return np.concatenate([obs, goal]), rew, done, info


class ResetWrapper(gym.Wrapper):
    def __init__(self, env, qpos, qvel, alpha=0.5):
        super().__init__(env)
        self.env = env
        self.alpha = alpha
        self.qpos = qpos
        self.qvel = qvel
        self.t = 0
        self.max_t = 1000
        print(PrintColors.BOLD + f"RegularReset: {self.alpha=}" + PrintColors.ENDC)

    def reset(self, seed=0, options=None):
        self.env.reset()
        if np.random.uniform() < self.alpha:
            idx = np.random.choice(len(self.qpos))
            t = np.random.choice(min(len(self.qpos[idx]), self.max_t))
            self.env.unwrapped.set_state(self.qpos[idx][t], self.qvel[idx][t])
            self.t = t
        else:
            self.t = 0
        return self.env.unwrapped._get_obs()

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        self.t += 1
        if self.t >= self.max_t:
            done = True
        return next_obs, rew, done, info

    def update_alpha(self, alpha):
        self.alpha = alpha
