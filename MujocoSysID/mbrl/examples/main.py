# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
import mbrl.algorithms.lamps as lamps
import mbrl.algorithms.sysid as sysid
# import mbrl.algorithms.mm as mm

import mbrl.util.env


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "pets":
        return pets.train(env, term_fn, reward_fn, cfg, silent=cfg.silent)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg, silent=cfg.silent)
    if cfg.algorithm.name == "planet":
        return planet.train(env, cfg)
    if cfg.algorithm.name == "lamps":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return lamps.train(env, test_env, term_fn, cfg, silent=cfg.silent)
    if cfg.algorithm.name == "sysid":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return sysid.train(env, test_env, term_fn, cfg, silent=cfg.silent)


if __name__ == "__main__":
    run()
