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
from mbrl.util.common import PrintColors
# import mbrl.algorithms.mm as mm

import mbrl.util.env


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    print(f"{PrintColors.BOLD}Config:")
    cfgs = {
        "use_yuda_default": cfg.use_yuda_default,
        "add_exp_to_replay_buffer": cfg.add_exp_to_replay_buffer,
        "use_policy_buffer_adv_update": cfg.use_policy_buffer_adv_update,
        "model_exp_ratio": cfg.overrides.model_exp_ratio,
        "policy_exp_ratio": cfg.overrides.policy_exp_ratio,
        "use_mbrl_demos": cfg.use_mbrl_demos,
        "sac_schedule_lr": cfg.sac_schedule_lr,
        "adversarial_reward_loss": cfg.adversarial_reward_loss,
    }
    for k, v in cfgs.items():
        if v != 0.0 and v is not False:
            print(f"{k}: {v}")

    print(f"seed: {cfg.seed}")
    print(
        f"Making {cfg.overrides.num_steps / cfg.eval_frequency} evaluations{PrintColors.ENDC}"
    )

    if cfg.debug_mode:
        print(PrintColors.WARNING + "Running in debug mode" + PrintColors.ENDC)

    env, term_fn, reward_fn = mbrl.util.env.EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.algorithm.name == "pets":
        return pets.train(env, term_fn, reward_fn, cfg, silent=cfg.silent)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg, silent=cfg.silent)
    if cfg.algorithm.name == "planet":
        return planet.train(env, cfg, silent=cfg.silent)
    if cfg.algorithm.name == "lamps":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return lamps.train(env, test_env, term_fn, cfg, silent=cfg.silent)
    if cfg.algorithm.name == "sysid":
        test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
        return sysid.train(env, test_env, term_fn, cfg, silent=cfg.silent)


if __name__ == "__main__":
    run()
