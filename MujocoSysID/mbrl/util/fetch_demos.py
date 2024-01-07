import numpy as np
import torch
import d4rl
import gym
import h5py
import copy
from pathlib import Path
import pdb

EPS = 1e-6


def fetch_demos(env_name, zero_out_rewards=True):
    env_name = env_name.replace("gym___", "")
    if "truncated" in env_name.lower():
        env_name = f"{env_name.split('_')[0].capitalize()}-v3"
    if "maze" in env_name:
        e = gym.make(env_name)
        dataset = e.get_dataset()
        term_raw = np.argwhere(
            np.logical_or(dataset["timeouts"] > 0, dataset["terminals"] > 0)
        )
        # filter out all one-step trajectories
        term = []
        for i in range(len(term_raw)):
            if term_raw[i][0] == 0:
                continue
            if term_raw[i][0] - term_raw[i - 1][0] > 1:
                term.append(term_raw[i])
        Js = []
        ranges = []
        start = 0
        for i in range(len(term)):
            ranges.append((start, term[i][0] + 1))
            J = dataset["rewards"][start : term[i][0] + 1].sum()
            Js.append(J)
            start = term[i][0] + 1
        Js = np.array(Js)
        exp_ranges = np.array(ranges)
        acts = np.concatenate(
            [
                dataset["actions"][exp_range[0] : exp_range[1] - 1]
                for exp_range in exp_ranges
            ]
        )
        obs = np.concatenate(
            [
                dataset["observations"][exp_range[0] : exp_range[1] - 1]
                for exp_range in exp_ranges
            ]
        )
        next_obs = np.concatenate(
            [
                dataset["observations"][exp_range[0] + 1 : exp_range[1]]
                for exp_range in exp_ranges
            ]
        )
        rewards = np.concatenate(
            [
                dataset["rewards"][exp_range[0] : exp_range[1] - 1]
                for exp_range in exp_ranges
            ]
        )
        terminals = np.concatenate(
            [
                dataset["terminals"][exp_range[0] : exp_range[1] - 1]
                for exp_range in exp_ranges
            ]
        )
        timeouts = np.concatenate(
            [
                dataset["timeouts"][exp_range[0] : exp_range[1] - 1]
                for exp_range in exp_ranges
            ]
        )
        goals = np.array(
            [
                dataset["infos/goal"][exp_range[0] : exp_range[1] - 1]
                for exp_range in exp_ranges
            ]
        )
        goals_flattened = np.array([g for traj in goals for g in traj])
        obs = np.concatenate([obs, goals_flattened], axis=1)
        next_obs = np.concatenate([next_obs, goals_flattened], axis=1)
        dataset = dict(
            observations=obs,
            actions=acts,
            next_observations=next_obs,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
        )
        dataset_path = "d4rl"
    else:
        num_demos, T = 64, 1000
        if "guest" in str(Path(Path.cwd())):
            project_root = Path("/home/guest/dev/juntao/")
        else:
            project_root = Path("/share/portal/jlr429/pessimistic-irl/")
        dataset_path = Path(project_root, "expert_data", f"{env_name}_100000_sb3.h5")
        dataset = h5py.File(dataset_path, "r")
        dataset = {
            key: np.array(dataset[key])[: num_demos * T] for key in dataset.keys()
        }

        term = np.argwhere(
            np.logical_or(dataset["timeouts"] > 0, dataset["terminals"] > 0)
        )
        Js = []
        ranges = []
        start = 0
        for i in range(len(term)):
            ranges.append((start, term[i][0] + 1))
            J = dataset["rewards"][start : term[i][0] + 1].sum()
            Js.append(J)
            start = term[i][0] + 1
        Js = np.array(Js)
        exp_ranges = np.array(ranges)

        for key in [
            "observations",
            "actions",
            "next_observations",
            "rewards",
            "terminals",
        ]:
            dataset[key] = np.concatenate(
                [dataset[key][exp_range[0] : exp_range[1]] for exp_range in exp_ranges]
            )
            if key == "actions":
                dataset[key] = np.clip(
                    dataset[key], -1 + EPS, 1 - EPS
                )  # due to tanh in TD3

        if "ant" in env_name.lower() or "humanoid" in env_name.lower():
            # get qpos and qvel dimensions
            print(f"Old dataset shape: {dataset['observations'].shape}")
            env = gym.make(env_name)
            qpos, qvel = (
                env.sim.data.qpos.ravel().copy(),
                env.sim.data.qvel.ravel().copy(),
            )
            qpos_dim, qvel_dim = qpos.shape[0], qvel.shape[0]
            obs_dim = (
                qpos_dim + qvel_dim - 2
            )  # truncated obs ignores first 2 elements of qpos
            dataset["observations"] = dataset["observations"][:, :obs_dim]
            dataset["next_observations"] = dataset["next_observations"][:, :obs_dim]
            print(f"New dataset shape: {dataset['observations'].shape}")

    if zero_out_rewards:
        dataset["rewards"] = np.zeros_like(dataset["rewards"])

    print("-" * 80)
    print(f"{dataset_path=}")
    print(f"{dataset.keys()=}")
    print(
        f"{np.mean(dataset['rewards'])=}\t{np.mean(Js)=}\t{np.std(Js)=}\t{zero_out_rewards=}"
    )
    print("-" * 80)

    return dataset
