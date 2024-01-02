# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, Sequence, cast

import gym
import hydra.utils
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
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder
from mbrl.util.fetch_demos import fetch_demos
from mbrl.models.discriminator import Discriminator
from mbrl.util.oadam import OAdam
from mbrl.util.common import gradient_penalty

import d4rl
from tqdm import tqdm

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]


class PrintColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
):
    batch = replay_buffer.sample(batch_size)
    initial_obs, *_ = cast(mbrl.types.TransitionBatch, batch).astuple()
    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    for i in range(rollout_horizon):
        action = agent.act(obs, sample=sac_samples_action, batched=True)
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        sac_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()


def evaluate(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
    video_recorder: VideoRecorder,
    maze=False,
) -> float:
    avg_episode_reward = 0
    success = 0
    for episode in range(num_episodes):
        obs = env.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            video_recorder.record(env)
            episode_reward += reward
        if maze:
            success += episode_reward > 0
            avg_episode_reward += episode_reward
        else:
            avg_episode_reward += episode_reward
    if maze:
        return avg_episode_reward / num_episodes, success / num_episodes
    return avg_episode_reward / num_episodes


def sample(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
) -> float:
    states, actions = [], []
    env_steps = 0
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            states.append(obs)
            action = agent.act(obs)
            actions.append(action)
            obs, _, done, _ = env.step(action)
            env_steps += 0
    return (
        torch.from_numpy(np.array(states)),
        torch.from_numpy(np.array(actions)),
        env_steps,
    )


def maybe_replace_sac_buffer(
    sac_buffer: Optional[mbrl.util.ReplayBuffer],
    obs_shape: Sequence[int],
    act_shape: Sequence[int],
    new_capacity: int,
    seed: int,
) -> mbrl.util.ReplayBuffer:
    if sac_buffer is None or new_capacity != sac_buffer.capacity:
        if sac_buffer is None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = sac_buffer.rng
        new_buffer = mbrl.util.ReplayBuffer(new_capacity, obs_shape, act_shape, rng=rng)
        if sac_buffer is None:
            return new_buffer
        obs, action, next_obs, reward, done = sac_buffer.get_all().astuple()
        new_buffer.add_batch(obs, action, next_obs, reward, done)
        return new_buffer
    return sac_buffer


def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    is_maze = "maze" in cfg.overrides.env
    expert_dataset = fetch_demos(
        cfg.overrides.env, zero_out_rewards=cfg.train_discriminator
    )

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = mbrl.util.Logger(work_dir, enable_back_compatible=True)
    logger.register_group(
        mbrl.constants.RESULTS_LOG_NAME,
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )
    save_video = cfg.get("save_video", False)
    video_recorder = VideoRecorder(work_dir if save_video else None)

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    policy_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    random_explore = cfg.algorithm.random_initial_explore
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
        additional_buffer=policy_buffer,
    )

    # ------------ Fill expert buffer ---------------------

    expert_replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    replay_buffer.add_batch(
        expert_dataset["observations"][:1000],
        expert_dataset["actions"][:1000],
        expert_dataset["next_observations"][:1000],
        expert_dataset["rewards"][:1000],
        expert_dataset["terminals"][:1000],
    )
    if cfg.from_end:
        print(f"{PrintColors.OKBLUE}Adding from end of expert dataset")
        expert_replay_buffer.add_batch(
            expert_dataset["observations"][-cfg.overrides.expert_size :],
            expert_dataset["actions"][-cfg.overrides.expert_size :],
            expert_dataset["next_observations"][-cfg.overrides.expert_size :],
            expert_dataset["rewards"][-cfg.overrides.expert_size :],
            expert_dataset["terminals"][-cfg.overrides.expert_size :],
        )
    else:
        print(f"{PrintColors.OKBLUE}Adding from end of expert dataset")
        expert_replay_buffer.add_batch(
            expert_dataset["observations"][: cfg.overrides.expert_size],
            expert_dataset["actions"][: cfg.overrides.expert_size],
            expert_dataset["next_observations"][: cfg.overrides.expert_size],
            expert_dataset["rewards"][: cfg.overrides.expert_size],
            expert_dataset["terminals"][: cfg.overrides.expert_size],
        )
    print(f"Expert buffer size: {cfg.overrides.expert_size}{PrintColors.ENDC}")

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    rollout_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    if cfg.train_discriminator:
        print(
            f"{PrintColors.OKBLUE}Training with discriminator function{PrintColors.ENDC}"
        )
        f_net = Discriminator(env).to(cfg.device)
        f_opt = OAdam(f_net.parameters(), lr=cfg.disc.lr)
        model_env = mbrl.models.ModelEnv(
            env, dynamics_model, termination_fn, f_net, generator=torch_generator
        )
        agent.sac_agent.add_f_net(f_net)
    else:
        print(
            f"{PrintColors.OKBLUE}Training with ground truth rewards{PrintColors.ENDC}"
        )
        model_env = mbrl.models.ModelEnv(
            env, dynamics_model, termination_fn, None, generator=torch_generator
        )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=None if silent else logger,
    )
    best_eval_reward = -np.inf
    epoch = 0
    disc_steps = 0
    sac_buffer = None
    tbar = tqdm(range(cfg.overrides.num_steps), ncols=0)
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            mbrl.util.math.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        sac_buffer_capacity = rollout_length * rollout_batch_size * trains_per_epoch
        sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
        sac_buffer = maybe_replace_sac_buffer(
            sac_buffer, obs_shape, act_shape, sac_buffer_capacity, cfg.seed
        )
        obs, done = None, False

        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or done:
                obs, done = env.reset(), False
            # --- Doing env step and adding to model dataset ---
            next_obs, reward, done, _ = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer, policy_buffer
            )

            (
                exp_obs,
                exp_next_obs,
                exp_act,
                exp_reward,
                exp_done,
            ) = expert_replay_buffer.sample_one()
            replay_buffer.add(exp_obs, exp_act, exp_next_obs, exp_reward, exp_done)

            # --------------- Model Training -----------------
            if (env_steps + 1) % int(cfg.overrides.freq_train_model / 2) == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                )

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together

                use_expert_data = rng.random() < cfg.overrides.model_exp_ratio
                # ! most configs have model_exp_ratio == 0.0, so use_expert_data is always False
                # ! however, replay_buffer contains a bit of expert data each iteration
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    expert_replay_buffer if use_expert_data else replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                )

                if cfg.update_with_model:
                    print("UPDATE WITH MODEL")
                    if not disc_steps == 0:
                        learning_rate_used = cfg.disc.lr / disc_steps
                    else:
                        learning_rate_used = cfg.disc.lr
                    f_opt = OAdam(f_net.parameters(), lr=learning_rate_used)

                    S_curr, A_curr, s = sample(
                        test_env, agent, cfg.disc.num_traj_samples
                    )
                    learner_sa_pairs = torch.cat((S_curr, A_curr), dim=1).to(cfg.device)
                    env_steps += s
                    tbar.update(s)
                    for _ in range(cfg.disc.num_updates_per_step):
                        learner_sa = learner_sa_pairs[
                            np.random.choice(len(learner_sa_pairs), cfg.disc.batch_size)
                        ]
                        expert_batch = expert_replay_buffer.sample(cfg.disc.batch_size)
                        expert_s, expert_a, *_ = cast(
                            mbrl.types.TransitionBatch, expert_batch
                        ).astuple()
                        expert_sa = torch.cat(
                            (torch.from_numpy(expert_s), torch.from_numpy(expert_a)),
                            dim=1,
                        ).to(cfg.device)
                        f_opt.zero_grad()
                        f_learner = f_net(learner_sa.float())
                        f_expert = f_net(expert_sa.float())
                        gp = gradient_penalty(learner_sa, expert_sa, f_net)
                        loss = f_expert.mean() - f_learner.mean() + 10 * gp
                        loss.backward()
                        f_opt.step()
                    disc_steps += 1

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                use_real_data = rng.random() < cfg.algorithm.real_data_ratio
                # ! in the existing configs, use_real_data is always False because real_data_ratio == 0.0
                which_buffer = replay_buffer if use_real_data else sac_buffer
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    which_buffer
                ) < cfg.overrides.sac_batch_size:
                    break  # only update every once in a while

                if cfg.overrides.policy_exp_ratio > 1:
                    agent.sac_agent.update_parameters(
                        which_buffer,
                        cfg.overrides.sac_batch_size,
                        updates_made,
                        logger,
                        reverse_mask=True,
                    )

                else:
                    if rng.random() < cfg.overrides.policy_exp_ratio:
                        agent.sac_agent.adv_update_parameters(
                            which_buffer,
                            expert_replay_buffer,
                            cfg.overrides.sac_batch_size,
                            updates_made,
                            logger,
                            reverse_mask=True,
                        )

                    else:
                        agent.sac_agent.adv_update_parameters(
                            which_buffer,
                            policy_buffer,
                            cfg.overrides.sac_batch_size,
                            updates_made,
                            logger,
                            reverse_mask=True,
                        )

                updates_made += 1
                if not silent and updates_made % cfg.log_frequency_agent == 0:
                    logger.dump(updates_made, save=True)

            # ------ Epoch ended (evaluate and save model) ------
            if (env_steps + 1) % cfg.overrides.epoch_length == 0:
                if not is_maze:
                    avg_reward = evaluate(
                        test_env,
                        agent,
                        cfg.algorithm.num_eval_episodes,
                        video_recorder,
                        is_maze,
                    )
                    logger.log_data(
                        mbrl.constants.RESULTS_LOG_NAME,
                        {
                            "epoch": epoch,
                            "env_step": env_steps,
                            "episode_reward": avg_reward,
                            "rollout_length": rollout_length,
                        },
                    )
                else:
                    avg_reward, success_rate = evaluate(
                        test_env,
                        agent,
                        cfg.algorithm.num_eval_episodes,
                        video_recorder,
                        is_maze,
                    )
                    logger.log_data(
                        mbrl.constants.RESULTS_LOG_NAME,
                        {
                            "epoch": epoch,
                            "env_step": env_steps,
                            "episode_reward": avg_reward,
                            "success_rate": success_rate,
                            "rollout_length": rollout_length,
                        },
                    )
                if avg_reward > best_eval_reward:
                    video_recorder.save(f"{epoch}.mp4")
                    best_eval_reward = avg_reward
                    agent.sac_agent.save_checkpoint(
                        ckpt_path=os.path.join(work_dir, "sac.pth")
                    )
                epoch += 1

            tbar.update(1)
            env_steps += 1
            obs = next_obs

        # ------ Discriminator Training ------
        if cfg.update_with_model:
            continue
        if cfg.train_discriminator and updates_made % cfg.disc.freq_train_disc == 0:
            print("HEREEE")
            if not disc_steps == 0:
                learning_rate_used = cfg.disc.lr / disc_steps
            else:
                learning_rate_used = cfg.disc.lr
            f_opt = OAdam(f_net.parameters(), lr=learning_rate_used)

            S_curr, A_curr, s = sample(test_env, agent, cfg.disc.num_traj_samples)
            learner_sa_pairs = torch.cat((S_curr, A_curr), dim=1).to(cfg.device)
            env_steps += s
            tbar.update(s)
            for _ in range(cfg.disc.num_updates_per_step):
                learner_sa = learner_sa_pairs[
                    np.random.choice(len(learner_sa_pairs), cfg.disc.batch_size)
                ]
                expert_batch = expert_replay_buffer.sample(cfg.disc.batch_size)
                expert_s, expert_a, *_ = cast(
                    mbrl.types.TransitionBatch, expert_batch
                ).astuple()
                expert_sa = torch.cat((expert_s, expert_a), dim=1).to(cfg.device)
                f_opt.zero_grad()
                f_learner = f_net(learner_sa.float())
                f_expert = f_net(expert_sa.float())
                gp = gradient_penalty(learner_sa, expert_sa, f_net)
                loss = f_expert.mean() - f_learner.mean() + 10 * gp
                loss.backward()
                f_opt.step()
            disc_steps += 1

    return np.float32(best_eval_reward)
