import numpy as np


class DiscriminatorReplayBuffer:
    def __init__(self, obs_space_size, action_space_size):
        self.obs_size = obs_space_size
        self.act_size = action_space_size
        self.size = 0
        self.obs = None
        self.actions = None
        self.first_addition = True

    def __len__(self):
        return self.size

    def size(self):
        return self.size

    def add(self, obs, act):
        # assumes obs and act are of type numpy
        if not len(obs[0]) == self.obs_size or not len(act[0]) == self.act_size:
            raise Exception("incoming samples do not match the correct size")
        if self.first_addition:
            self.first_addition = False
            self.obs = np.array(obs)
            self.actions = np.array(act)
        else:
            self.obs = np.append(self.obs, obs, axis=0)
            self.actions = np.append(self.actions, act, axis=0)
        self.size += len(obs)
        return

    def sample(self, batch):
        indexes = np.random.choice(range(self.size), batch)
        return self.obs[indexes], self.actions[indexes]
