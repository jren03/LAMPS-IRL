# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import omegaconf
import torch
import gym
from termcolor import cprint

import mbrl.algorithms.lamps_am as lamps_am
import mbrl.algorithms.model_pretraining as mp
import mbrl.util.env

from pygit2 import Repository


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env_name = cfg.overrides.env.lower().replace("gym___", "")
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)

    env = gym.make(env_name)
    test_env = gym.make(env_name)

    if cfg.train_discriminator:
        cprint("Training discriminator", color="green", attrs=["bold"])
        if cfg.use_ensemble:
            cprint("Using ensemble", color="green", attrs=["bold"])
        else:
            cprint("Not using ensemble", color="green", attrs=["bold"])
    else:
        cprint("Using ground truth", color="green", attrs=["bold"])

    cprint(f"seed: {cfg.seed}", color="green", attrs=["bold"])
    cprint(
        f"ema: {cfg.ema}\tpsdp_wrapper: {cfg.psdp_wrapper}\ttanh_disc: {cfg.tanh_disc}",
        color="green",
        attrs=["bold"],
    )
    cprint(
        f"clip_md: {cfg.clip_md}\twd_md: {cfg.wd_md}\tdecay_lr: {cfg.decay_lr}",
        color="green",
        attrs=["bold"],
    )

    branch_name = Repository(".").head.shorthand
    cprint("On branch: " + branch_name, "magenta", attrs=["bold"])

    if cfg.algorithm.name == "lamps_am":
        return lamps_am.train(env, test_env, term_fn, cfg, silent=cfg.silent)
    elif cfg.algorithm.name == "model_pretraining":
        return mp.train(env, cfg)
    else:
        raise NotImplementedError
    # if cfg.algorithm.name == "pets":
    #     return pets.train(env, term_fn, reward_fn, cfg, silent=cfg.silent)
    # if cfg.algorithm.name == "mbpo":
    #     test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
    #     return mbpo.train(env, test_env, term_fn, cfg, silent=cfg.silent)
    # if cfg.algorithm.name == "planet":
    #     return planet.train(env, cfg)
    # if cfg.algorithm.name == "lamps":
    #     test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
    #     return lamps.train(env, test_env, term_fn, cfg, silent=cfg.silent)
    # if cfg.algorithm.name == "sysid":
    #     test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
    #     return sysid.train(env, test_env, term_fn, cfg, silent=cfg.silent)


if __name__ == "__main__":
    run()
