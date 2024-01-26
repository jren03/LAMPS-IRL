import tqdm
import d4rl
import gym
import h5py
import torch
import argparse
import numpy as np
from pathlib import Path


if "guest" in str(Path(Path.cwd())):
    data_root = Path("/home/guest/dev/juntao/expert_data")
else:
    data_root = Path("/share/portal/jlr429/pessimistic-irl/expert_data")


def fetch_demos(env_name, cfg=None):
    env_name = env_name.replace("gym___", "")

    possible_data_path = Path(data_root, f"{env_name}_psdp_demos.npz")
    if possible_data_path.exists():
        print(f"Loading from {possible_data_path}")
        data = np.load(possible_data_path, allow_pickle=True)
        dataset = {
            "observations": data["observations"],
            "actions": data["actions"],
            "next_observations": data["next_observations"],
            "rewards": data["rewards"],
            "terminals": data["terminals"],
        }
        if cfg.train_discriminator:
            dataset["rewards"] = np.zeros_like(data["rewards"])
        print(f"{np.mean(dataset['rewards'])=}")
        return (
            dataset,
            torch.from_numpy(data["expert_sa_pairs"]),
            data["qpos"],
            data["qvel"],
            data["goals"],
            data["expert_reset_states"],
        )

    if "maze" in env_name:
        env = gym.make(env_name)
        dataset = env.get_dataset()
        term = np.argwhere(
            np.logical_or(dataset["timeouts"] > 0, dataset["terminals"] > 0)
        )
        start = 0
        expert_ranges = []
        for i in range(len(term)):
            expert_ranges.append([start, term[i][0] + 1])
            start = term[i][0] + 1
        expert_ranges = np.array(expert_ranges)
        obs_ref = {}
        for exp_range in expert_ranges:
            for i, obs in enumerate(
                dataset["observations"][exp_range[0] : exp_range[1]]
            ):
                # add timestep
                obs_ref[tuple(obs)] = i

        q_dataset = d4rl.qlearning_dataset(env)
        curr_obs_pt = 0
        curr_obs_indices = []
        next_obs_pt = 0
        next_obs_indices = []
        for i in tqdm.tqdm(range(len(q_dataset["observations"]))):
            while (
                np.linalg.norm(
                    q_dataset["observations"][i] - dataset["observations"][curr_obs_pt]
                )
                > 1e-10
            ):
                curr_obs_pt += 1
            curr_obs_indices.append(curr_obs_pt)
            while (
                np.linalg.norm(
                    q_dataset["next_observations"][i]
                    - dataset["observations"][next_obs_pt]
                )
                > 1e-10
            ):
                next_obs_pt += 1
            next_obs_indices.append(next_obs_pt)
        curr_obs_indices = np.array(curr_obs_indices)
        goals_flat = dataset["infos/goal"][curr_obs_indices]
        qpos_flat = dataset["infos/qpos"][curr_obs_indices]
        qvel_flat = dataset["infos/qvel"][curr_obs_indices]
        next_obs_indices = np.array(next_obs_indices)
        next_goals_flat = dataset["infos/goal"][next_obs_indices]

        observations = np.concatenate([q_dataset["observations"], goals_flat], axis=1)
        next_observations = np.concatenate(
            [q_dataset["next_observations"], next_goals_flat], axis=1
        )
        rewards = q_dataset["rewards"]
        terminals = q_dataset["terminals"]
        actions = q_dataset["actions"]

        # add timestep to observations in q_dataset and dataset
        new_observations = np.zeros((observations.shape[0], observations.shape[1] + 1))
        new_next_observations = np.zeros(
            (next_observations.shape[0], next_observations.shape[1] + 1)
        )
        for i, (obs_g, next_obs_g) in enumerate(zip(observations, next_observations)):
            new_observations[i] = np.concatenate(
                (
                    observations[i],
                    np.array([obs_ref[tuple(obs_g[: -len(goals_flat[0])])]]),
                ),
                axis=0,
            )
            new_next_observations[i] = np.concatenate(
                (
                    next_observations[i],
                    np.array([obs_ref[tuple(next_obs_g[: -len(goals_flat[0])])]]),
                ),
                axis=0,
            )

        new_dataset = {
            "observations": new_observations,
            "actions": actions,
            "next_observations": new_next_observations,
            "rewards": np.zeros_like(rewards),
            "terminals": terminals,
        }

        term = np.argwhere(terminals.flatten() > 0)
        start = 0
        qpos, qvel, goals = [], [], []
        for i in range(len(term)):
            qpos.append(qpos_flat[start : term[i][0] + 1])
            qvel.append(qvel_flat[start : term[i][0] + 1])
            goals.append(goals_flat[start : term[i][0] + 1])
            start = term[i][0] + 1

        obs_goal_cat = np.zeros(
            (
                dataset["observations"].shape[0],
                dataset["observations"].shape[1] + dataset["infos/goal"].shape[1] + 1,
            )
        )
        for obs, goal in zip(dataset["observations"], dataset["infos/goal"]):
            obs_goal_cat[i] = np.concatenate(
                (obs, goal, np.array([obs_ref.get(tuple(obs), 0)])),
                axis=0,
            )

        expert_reset_states = np.array(
            [
                obs_goal_cat[expert_ranges[i][0] : expert_ranges[i][1]]
                for i in range(len(expert_ranges))
            ]
        )

        expert_reset_states_square = []
        max_length = max(len(row) for row in expert_reset_states)
        for i in range(len(expert_reset_states)):
            if len(expert_reset_states[i]) == max_length:
                expert_reset_states_square.append(expert_reset_states[i])
                continue
            # duplicate each state until we reach max_length
            repeat_times = max_length // len(expert_reset_states[i])
            remainder = max_length % len(expert_reset_states[i])
            if remainder > 0:
                expert_reset_states[i] = np.concatenate(
                    (
                        np.repeat(
                            expert_reset_states[i][:remainder], repeat_times + 1, axis=0
                        ),
                        np.repeat(
                            expert_reset_states[i][remainder:], repeat_times, axis=0
                        ),
                    )
                )
            else:
                expert_reset_states[i] = np.repeat(
                    expert_reset_states[i], repeat_times, axis=0
                )
            assert len(expert_reset_states[i]) == max_length
            expert_reset_states_square.append(expert_reset_states[i])

        expert_reset_states = np.array(expert_reset_states_square)
        expert_sa_pairs = torch.cat(
            (torch.from_numpy(new_observations), torch.from_numpy(actions)), dim=1
        )
    else:
        raise NotImplementedError

    return new_dataset, expert_sa_pairs, qpos, qvel, goals, expert_reset_states
