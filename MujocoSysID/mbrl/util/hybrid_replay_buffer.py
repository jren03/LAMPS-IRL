from gym import spaces
import numpy as np
from typing import Dict, Generator, Optional, Union
import torch as th


from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer


class HybridReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        expert_data: dict = dict(),
        balanced_sampling: bool = False,
        fixed_hybrid_schedule: bool = False,
    ):
        super(HybridReplayBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
        )
        self.expert_states = expert_data["observations"]
        self.expert_actions = expert_data["actions"]
        self.expert_next_states = expert_data["next_observations"]
        self.expert_dones = expert_data["terminals"]
        self.expert_rewards = expert_data["rewards"]
        self.expert_timeouts = expert_data["timeouts"]

        # self.expert_states = self.normalize_expert_obs(self.expert_states)
        # self.expert_next_states = self.normalize_expert_obs(self.expert_next_states)

        self.balanced_sampling = balanced_sampling
        self.fixed_hybrid_schedule = fixed_hybrid_schedule
        self.offline_schedule = np.array([[0.2, 0.2, 300000], [0.2, 0.1, int(1e6)]])
        self.steps = 0
        self.ratio_lag = 0
        if self.balanced_sampling:
            print("=============== BALANCED SAMPLING =================")
            print(f"=============== {self.offline_schedule} =================")
        else:
            print("=============== REGULAR SAMPLING =================")

    def _get_ratio(self, t):
        if self.fixed_hybrid_schedule:
            return self.offline_schedule[0, 0]
        if t > self.offline_schedule[0, 2] and self.offline_schedule.shape[0] > 1:
            self.offline_schedule = np.delete(self.offline_schedule, 0, 0)
            self.ratio_lag = t
        max_lr, min_lr, lr_steps = self.offline_schedule[0]
        ratio = max_lr - min(1, (t - self.ratio_lag) / (lr_steps - self.ratio_lag)) * (
            max_lr - min_lr
        )
        self.steps += 1
        return ratio

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        num_samples = len(batch_inds)
        if self.balanced_sampling:
            offline_ratio = self._get_ratio(self.steps)
            num_expert_samples = int(num_samples * offline_ratio)
            num_learner_samples = int(num_samples * (1 - offline_ratio))
            # if self.balanced_smapling is not None:
            #     num_expert_samples = int(round(num_samples * self.balanced_percentage))
            #     num_learner_samples = int(
            #         round(num_samples * (1 - self.balanced_percentage))
            #     )
            #     assert (
            #         num_learner_samples + num_expert_samples == num_samples
            #     ), f"{self.balanced_percentage=}, {num_learner_samples=} + {num_expert_samples=} != {num_samples=}"
            # else:
            #     num_expert_samples = int(num_samples / 2)
            #     num_learner_samples = int(num_samples / 2)

            learner_inds = batch_inds[:num_learner_samples]
            expert_inds = np.random.randint(
                0, len(self.expert_states), size=num_expert_samples, dtype=int
            )

            if self.optimize_memory_usage:
                next_obs = self._normalize_obs(
                    self.observations[(learner_inds + 1) % self.buffer_size, 0, :],
                    env,
                )
            else:
                next_obs = self._normalize_obs(
                    self.next_observations[learner_inds, 0, :], env
                )

            obs = self._normalize_obs(self.observations[learner_inds, 0, :], env)
            actions = self.actions[learner_inds, 0, :]
            dones = self.dones[learner_inds]
            # timeouts = self.timeouts[learner_inds]
            rewards = self.rewards[learner_inds]

            if expert_inds.shape[0] > 0:
                next_obs = np.concatenate(
                    (
                        next_obs,
                        self._normalize_obs(self.expert_next_states[expert_inds], env),
                    ),
                    axis=0,
                )
                obs = np.concatenate(
                    (obs, self._normalize_obs(self.expert_states[expert_inds], env)),
                    axis=0,
                )
                actions = np.concatenate(
                    (
                        actions,
                        self.expert_actions[expert_inds].reshape(
                            num_expert_samples, -1
                        ),
                    ),
                    axis=0,
                )
                dones = np.concatenate(
                    (
                        dones,
                        self.expert_dones[expert_inds].reshape(num_expert_samples, -1),
                    ),
                    axis=0,
                )
                # timeouts = np.concatenate(
                #     (
                #         timeouts,
                #         self.expert_timeouts[expert_inds].reshape(
                #             num_expert_samples, -1
                #         ),
                #     ),
                #     axis=0,
                # )
                rewards = np.concatenate(
                    (
                        rewards,
                        self.expert_rewards[expert_inds].reshape(
                            num_expert_samples, -1
                        ),
                    ),
                    axis=0,
                )

            data = (
                obs,
                actions,
                next_obs,
                (dones).reshape(-1, 1),
                self._normalize_reward(rewards.reshape(-1, 1), env),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
        else:
            env_indices = np.random.randint(
                0, high=self.n_envs, size=(len(batch_inds),)
            )
            if self.optimize_memory_usage:
                next_obs = self._normalize_obs(
                    self.observations[
                        (batch_inds + 1) % self.buffer_size, env_indices, :
                    ],
                    env,
                )
            else:
                next_obs = self._normalize_obs(
                    self.next_observations[batch_inds, env_indices, :], env
                )

            data = (
                self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
                self.actions[batch_inds, env_indices, :],
                next_obs,
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                (
                    self.dones[batch_inds, env_indices]
                    # * (1 - self.timeouts[batch_inds, env_indices])
                ).reshape(-1, 1),
                self._normalize_reward(
                    self.rewards[batch_inds, env_indices].reshape(-1, 1), env
                ),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def normalize_expert_obs(self, obs):
        def compute_mean_std(states: np.ndarray, eps: float):
            mean = states.mean(0)
            std = states.std(0) + eps
            return mean, std

        def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
            return (states - mean) / std

        state_mean, state_std = compute_mean_std(obs, eps=1e-3)
        return normalize_states(obs, state_mean, state_std)
