import tqdm
import d4rl
import gym
import h5py
import torch
import argparse
import numpy as np
from pathlib import Path


data_root = Path("/share/portal/jlr429/pessimistic-irl/expert_data")


def fetch_demos(env_name, zero_out_rewards=True, use_mbrl_demos=False):
    env_name = env_name.replace("gym___", "")
    possible_data_path = Path(data_root, f"{env_name}_demos.npz")
    if possible_data_path.exists():
        print(f"Loading from {possible_data_path}")
        data = np.load(possible_data_path, allow_pickle=True)
        print(f"{np.mean(data['rewards'])=}")
        return (
            {
                "observations": data["observations"],
                "actions": data["actions"],
                "next_observations": data["next_observations"],
                "rewards": data["rewards"],
                "terminals": data["terminals"],
            },
            torch.from_numpy(data["expert_sa_pairs"]),
            data["qpos"],
            data["qvel"],
            data["goals"],
        )

    if "maze" in env_name:
        env = gym.make(env_name)
        dataset = env.get_dataset()
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

        dataset = {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
            "rewards": rewards,
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

        expert_sa_pairs = torch.cat(
            (torch.from_numpy(observations), torch.from_numpy(actions)), dim=1
        )
    else:
        raise NotImplementedError

    return dataset, expert_sa_pairs, qpos, qvel, goals
