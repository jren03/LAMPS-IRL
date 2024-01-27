from pathlib import Path

from mbrl.util.fetch_demos import fetch_demos
from ast import Not
import os
from typing import Optional, Sequence, cast

import time
import gym
import hydra.utils
from networkx import minimum_node_cut
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
import functools
import itertools
from mbrl.planning.sac_wrapper import SACAgent

from tqdm import tqdm

import torch.nn as nn

from mbrl.env.gym_wrappers import (
    AntMazeResetWrapper,
    GoalWrapper,
    RewardWrapper,
    TremblingHandWrapper,
    PSDPWrapper,
)
from mbrl.util.fetch_demos import fetch_demos
from mbrl.util.psdp_fetch_demos import fetch_demos as psdp_fetch_demos
from mbrl.util.am_buffers import QReplayBuffer
from mbrl.models.arch import Discriminator, DiscriminatorEnsemble
from mbrl.models.td3_bc import TD3_BC
from mbrl.util.oadam import OAdam
from stable_baselines3.common.evaluation import evaluate_policy
from mbrl.util.nn_utils import gradient_penalty
from termcolor import cprint

from mbrl.util.ema_pytorch import EMA

import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gym.wrappers
import hydra
import numpy as np
import omegaconf

import mbrl.models
import mbrl.planning
import mbrl.types

from mbrl.util.replay_buffer import (
    BootstrapIterator,
    ReplayBuffer,
    TransitionIterator,
)
from mbrl.types import TransitionBatch
from collections import defaultdict

import sys
import os


def maybe_create_train_val_split(
    expert_dataset: Dict[str, np.ndarray],
    train_dataset_path: pathlib.Path,
    val_dataset_path: pathlib.Path,
    val_ratio: float = 0.2,
):
    if train_dataset_path.exists() and val_dataset_path.exists():
        cprint("Loading train/val split", "green")
        return
    else:
        cprint("Creating train/val split", "red")
    assert val_ratio < 1.0

    # Create a 2D histogram with binning
    x, y = expert_dataset["observations"][:, 0], expert_dataset["observations"][:, 1]
    bins = 50
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # Digitize the data points
    x_dig = np.digitize(x, xedges) - 1
    y_dig = np.digitize(y, yedges) - 1

    # Ensure the indices are within the valid range
    x_dig = np.clip(x_dig, 0, hist.shape[0] - 1)
    y_dig = np.clip(y_dig, 0, hist.shape[1] - 1)

    # Create a dictionary to store the indices of the data points in each bin
    bin_indices = defaultdict(list)
    for i, (bin_x, bin_y) in enumerate(zip(x_dig, y_dig)):
        bin_indices[(bin_x, bin_y)].append(i)

    # For each bin, sample a certain ratio of the data points
    val_indices = []
    for bin_data_indices in bin_indices.values():
        num_samples = int(len(bin_data_indices) * val_ratio)
        val_indices.extend(
            np.random.choice(bin_data_indices, size=num_samples, replace=False)
        )

    # Create a mask for the sampled data points
    sampled_mask = np.zeros(len(x), dtype=bool)
    sampled_mask[val_indices] = True

    # Create a dictionary to store the training and validation data
    train_dataset = {
        "observations": expert_dataset["observations"][~sampled_mask],
        "actions": expert_dataset["actions"][~sampled_mask],
        "next_observations": expert_dataset["next_observations"][~sampled_mask],
        "rewards": expert_dataset["rewards"][~sampled_mask],
        "terminals": expert_dataset["terminals"][~sampled_mask],
    }
    val_dataset = {
        "observations": expert_dataset["observations"][sampled_mask],
        "actions": expert_dataset["actions"][sampled_mask],
        "next_observations": expert_dataset["next_observations"][sampled_mask],
        "rewards": expert_dataset["rewards"][sampled_mask],
        "terminals": expert_dataset["terminals"][sampled_mask],
    }

    # Save dataset to path
    np.savez(str(train_dataset_path), **train_dataset)
    np.savez(str(val_dataset_path), **val_dataset)


def get_train_val_ters(
    train_dataset_path,
    val_dataset_path,
    batch_size,
    rng,
    ensemble_size,
    shuffle_each_epoch,
    bootstrap_permutes,
):
    train_dataset = np.load(str(train_dataset_path))
    val_dataset = np.load(str(val_dataset_path))
    dataset_train = TransitionBatch(
        train_dataset["observations"],
        train_dataset["actions"],
        train_dataset["next_observations"],
        train_dataset["rewards"],
        train_dataset["terminals"],
    )
    dataset_val = TransitionBatch(
        val_dataset["observations"],
        val_dataset["actions"],
        val_dataset["next_observations"],
        val_dataset["rewards"],
        val_dataset["terminals"],
    )
    train_iter = BootstrapIterator(
        dataset_train,
        batch_size,
        ensemble_size,
        shuffle_each_epoch=shuffle_each_epoch,
        permute_indices=bootstrap_permutes,
        rng=rng,
    )
    val_iter = TransitionIterator(
        dataset_val, batch_size, shuffle_each_epoch=False, rng=rng
    )

    return train_iter, val_iter


def create_env(env_name, psdp_wrapper: bool = False, f_net: Optional[nn.Module] = None):
    env = gym.make(env_name)
    if "maze" in env_name:
        env = GoalWrapper(env)
    else:
        raise NotImplementedError
    if psdp_wrapper:
        env = PSDPWrapper(env)
    if f_net is not None:
        env = RewardWrapper(env, f_net)
    return env


def get_basic_buffer_iterators(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    val_ratio: float,
    ensemble_size: int = 1,
    shuffle_each_epoch: bool = True,
    bootstrap_permutes: bool = False,
) -> Tuple[TransitionIterator, Optional[TransitionIterator]]:
    """Returns training/validation iterators for the data in the replay buffer.

    Args:
        replay_buffer (:class:`mbrl.util.ReplayBuffer`): the replay buffer from which
            data will be sampled.
        batch_size (int): the batch size for the iterators.
        val_ratio (float): the proportion of data to use for validation. If 0., the
            validation buffer will be set to ``None``.
        ensemble_size (int): the size of the ensemble being trained.
        shuffle_each_epoch (bool): if ``True``, the iterator will shuffle the
            order each time a loop starts. Otherwise the iteration order will
            be the same. Defaults to ``True``.
        bootstrap_permutes (bool): if ``True``, the bootstrap iterator will create
            the bootstrap data using permutations of the original data. Otherwise
            it will use sampling with replacement. Defaults to ``False``.

    Returns:
        (tuple of :class:`mbrl.replay_buffer.TransitionIterator`): the training
        and validation iterators, respectively.
    """
    data = replay_buffer.get_all(shuffle=True)
    val_size = int(replay_buffer.num_stored * val_ratio)
    train_size = replay_buffer.num_stored - val_size
    train_data = data[:train_size]
    train_iter = BootstrapIterator(
        train_data,
        batch_size,
        ensemble_size,
        shuffle_each_epoch=shuffle_each_epoch,
        permute_indices=bootstrap_permutes,
        rng=replay_buffer.rng,
    )

    val_iter = None
    if val_size > 0:
        val_data = data[train_size:]
        val_iter = TransitionIterator(
            val_data, batch_size, shuffle_each_epoch=False, rng=replay_buffer.rng
        )

    return train_iter, val_iter


def train(
    env: gym.Env,
    cfg: omegaconf.DictConfig,
):
    env_name = cfg.overrides.env.lower().replace("gym___", "")

    # ------------------------------ Logging/Seeding ------------------------------ #
    model_train_dir = Path(
        f"/share/portal/jlr429/pessimistic-irl/LAMPS-IRL/MujocoSysID/model_train_dir/{env_name}"
    )
    model_train_dir.mkdir(exist_ok=True)
    for csv in model_train_dir.glob("*.csv"):
        cprint(f"Unlinking {csv.stem}", "red", attrs=["bold"])
        csv.unlink()
    logger = mbrl.util.Logger(model_train_dir, enable_back_compatible=True)
    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # ------------------------------ Buffer/Env ------------------------------ #
    env = GoalWrapper(env)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    expert_dataset, *_ = fetch_demos(env_name, cfg)
    expert_replay_buffer = mbrl.util.replay_buffer.ReplayBuffer(
        capacity=len(expert_dataset["observations"]),
        obs_shape=obs_shape,
        action_shape=act_shape,
        rng=rng,
    )
    expert_replay_buffer.add_batch(
        expert_dataset["observations"],
        expert_dataset["actions"],
        expert_dataset["next_observations"],
        expert_dataset["rewards"],
        expert_dataset["terminals"],
    )

    # ------------------------------ Model ------------------------------ #
    model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    model_trainer = mbrl.models.ModelTrainer(
        model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    # should load from a predetermined split that I can visualize
    validation_ratio = cfg.overrides.validation_ratio
    train_ratio_int = int((1 - validation_ratio) * 100)
    validation_ratio_int = int(validation_ratio * 100)
    dataset_path = Path(
        model_train_dir, f"dataset_{train_ratio_int}_{validation_ratio_int}"
    )
    dataset_path.mkdir(exist_ok=True)
    train_dataset_path = Path(dataset_path, "train.npz")
    val_dataset_path = Path(dataset_path, "val.npz")
    maybe_create_train_val_split(
        expert_dataset=expert_dataset,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        val_ratio=validation_ratio,
    )
    dataset_train, dataset_val = get_train_val_ters(
        train_dataset_path,
        val_dataset_path,
        cfg.overrides.model_batch_size,
        rng=rng,
        ensemble_size=len(model),
        shuffle_each_epoch=True,
        bootstrap_permutes=cfg.get("bootstrap_permutes", False),
    )

    if hasattr(model, "update_normalizer"):
        model.update_normalizer(expert_replay_buffer.get_all())

    model_trainer.train(
        dataset_train,
        dataset_val=dataset_val,
        num_epochs=cfg.get("num_epochs_train_model", None),
        patience=cfg.get("patience", 1),
        improvement_threshold=cfg.get("improvement_threshold", 0.01),
    )
    if model_train_dir is not None:
        model.save(str(model_train_dir))
