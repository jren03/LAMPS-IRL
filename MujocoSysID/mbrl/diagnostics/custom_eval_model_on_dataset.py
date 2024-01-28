# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

import mbrl.util
import mbrl.util.common
import omegaconf

import d4rl
import gym
from mbrl.env.gym_wrappers import (
    GoalWrapper,
)


class DatasetEvaluator:
    def __init__(self, root_dir: str):
        root_dir = Path(root_dir)
        self.output_path = Path(root_dir, "eval_results")
        self.output_path.mkdir(parents=True, exist_ok=True)

        # for loading the config file specifically
        cfg_file = root_dir / ".hydra" / "config.yaml"
        self.cfg = omegaconf.OmegaConf.load(cfg_file)
        self.handler = mbrl.util.create_handler(self.cfg)

        env_name = self.cfg.overrides.env.lower().replace("gym___", "")
        self.env = gym.make(env_name)
        self.env = GoalWrapper(self.env)
        # self.env, term_fn, reward_fn = self.handler.make_env(self.cfg)
        # self.reward_fn = reward_fn

        self.dynamics_model = mbrl.util.common.create_one_dim_tr_model(
            self.cfg,
            self.env.observation_space.shape,
            self.env.action_space.shape,
            model_dir=root_dir,
        )

        dataset_path = Path(
            root_dir,
            "dataset_80_20",
            "val.npz",
        )
        val_dataset = np.load(dataset_path)

        obs_shape = self.env.observation_space.shape
        act_shape = self.env.action_space.shape
        rng = np.random.default_rng(seed=self.cfg.seed)
        self.replay_buffer = mbrl.util.replay_buffer.ReplayBuffer(
            capacity=len(val_dataset["observations"]),
            obs_shape=obs_shape,
            action_shape=act_shape,
            rng=rng,
        )
        self.replay_buffer.add_batch(
            val_dataset["observations"],
            val_dataset["actions"],
            val_dataset["next_observations"],
            val_dataset["rewards"],
            val_dataset["terminals"],
        )

        # self.replay_buffer = mbrl.util.common.create_replay_buffer(
        #     self.cfg,
        #     self.env.observation_space.shape,
        #     self.env.action_space.shape,
        #     load_dir=dataset_dir,
        # )

    def plot_dataset_results(self, dataset: mbrl.util.TransitionIterator):
        all_means: List[np.ndarray] = []
        all_targets = []

        # Iterating over dataset and computing predictions
        for batch in dataset:
            (
                outputs,
                target,
            ) = self.dynamics_model.get_output_and_targets(batch)

            all_means.append(outputs[0].cpu().numpy())
            all_targets.append(target.cpu().numpy())

        # Consolidating targets and predictions
        all_means_np = np.concatenate(all_means, axis=-2)
        targets_np = np.concatenate(all_targets, axis=0)

        if all_means_np.ndim == 2:
            all_means_np = all_means_np[np.newaxis, :]
        assert all_means_np.ndim == 3  # ensemble, batch, target_dim

        # Visualization
        num_dim = targets_np.shape[1]
        for dim in range(num_dim):
            sort_idx = targets_np[:, dim].argsort()
            subsample_size = len(sort_idx) // 20 + 1
            subsample = np.random.choice(len(sort_idx), size=(subsample_size,))
            means = all_means_np[..., sort_idx, dim][..., subsample]  # type: ignore
            target = targets_np[sort_idx, dim][subsample]

            plt.figure(figsize=(8, 8))
            for i in range(all_means_np.shape[0]):
                plt.plot(target, means[i], ".", markersize=2)
            mean_of_means = means.mean(0)
            mean_sort_idx = target.argsort()
            plt.plot(
                target[mean_sort_idx],
                mean_of_means[mean_sort_idx],
                color="r",
                linewidth=0.5,
            )
            plt.plot(
                [target.min(), target.max()],
                [target.min(), target.max()],
                linewidth=2,
                color="k",
            )
            plt.xlabel("Target")
            plt.ylabel("Prediction")
            fname = self.output_path / f"pred_dim{dim}.png"
            plt.savefig(fname)
            plt.close()

    def run(self):
        batch_size = 32
        if hasattr(self.dynamics_model, "set_propagation_method"):
            self.dynamics_model.set_propagation_method(None)
            # Some models (e.g., GaussianMLP) require the batch size to be
            # a multiple of number of models
            batch_size = len(self.dynamics_model) * 8
        dataset, _ = mbrl.util.common.get_basic_buffer_iterators(
            self.replay_buffer, batch_size=batch_size, val_ratio=0
        )

        self.plot_dataset_results(dataset)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_dir", type=str, default=None)
    # parser.add_argument("--dataset_dir", type=str, default=None)
    # parser.add_argument("--results_dir", type=str, default=None)
    # args = parser.parse_args()

    # if not args.dataset_dir:
    #     args.dataset_dir = args.model_dir
    # evaluator = DatasetEvaluator(args.model_dir, args.dataset_dir, args.results_dir)

    root_dir = "/share/portal/jlr429/pessimistic-irl/LAMPS-IRL/MujocoSysID/model_train_dir/antmaze-large-diverse-v2"
    evaluator = DatasetEvaluator(root_dir)

    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["font.size"] = 14

    evaluator.run()
