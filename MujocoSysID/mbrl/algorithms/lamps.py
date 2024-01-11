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
import pprint

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
from mbrl.models.discriminator import Discriminator, DiscriminatorEnsemble
from mbrl.util.oadam import OAdam
from mbrl.util.common import gradient_penalty, PrintColors
from mbrl.util.discriminator_replay_buffer import DiscriminatorReplayBuffer

import d4rl
from tqdm import tqdm

import stable_baselines3 as sb3
from pathlib import Path

MBPO_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
    ("sac_reset_ratio", "SRR", "float"),
]


def rollout_model_and_populate_sac_buffer(
    model_env: mbrl.models.ModelEnv,
    replay_buffer: mbrl.util.ReplayBuffer,
    agent: SACAgent,
    sac_buffer: mbrl.util.ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    batch_size: int,
    fixed_reward_value: bool = False,
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
        if fixed_reward_value:
            sac_buffer.add_batch(
                obs[~accum_dones],
                action[~accum_dones],
                pred_next_obs[~accum_dones],
                np.zeros_like(pred_rewards[~accum_dones, 0]),
                pred_dones[~accum_dones, 0],
            )
        else:
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
    replay_buffer: DiscriminatorReplayBuffer,
    no_regret=False,
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
            env_steps += 1
    states_np, actions_np = np.array(states), np.array(actions)
    if no_regret:
        replay_buffer.add(states_np, actions_np)
    return states_np, actions_np, env_steps


def sample_from_learned_model(
    env: gym.Env,
    model_env: mbrl.models.ModelEnv,
    agent: sb3.SAC,
    num_episodes: int,
    rollout_horizon: int,
):
    states, actions = [], []
    env_steps = 0
    for episode in range(num_episodes):
        real_env_obs = env.reset()
        real_env_obs = real_env_obs.reshape(1, -1)
        model_state = model_env.reset(
            initial_obs_batch=cast(np.ndarray, real_env_obs),
            return_as_np=True,
        )
        obs = real_env_obs
        for _ in range(rollout_horizon):
            breakpoint()
            action = agent.predict(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]
            states.append(obs)
            actions.append(action)
            obs, _, done, model_state = model_env.step(
                action, model_state, sample=False
            )
            env_steps += 1
            if done:
                break
    states_np, actions_np = np.array(states), np.array(actions)
    return states_np, actions_np, env_steps


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
    pp = pprint.PrettyPrinter(indent=4)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    mbrl.planning.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent))
    )

    if cfg.train_disc_in_model:
        # load in SB3 model used to collect data
        expert_base_path = Path(
            "/share/portal/jlr429/pessimistic-irl/fast_irl/experts/"
        )
        env_name = cfg.overrides.env.lower()
        if "humanoid" in env_name:
            env_name = "Humanoid-v3"
        elif "ant" in env_name and "truncated" in env_name:
            env_name = "Ant-v3"
        else:
            env_name = env_name.replace("gym___", "")
        expert_path = Path(expert_base_path, f"{env_name}/expert")
        expert_sb3_agent = sb3.SAC.load(str(expert_path))
        print(f"{PrintColors.BOLD}Loading expert agent from {expert_path}")

    is_maze = "maze" in cfg.overrides.env
    expert_dataset = fetch_demos(
        cfg.overrides.env,
        zero_out_rewards=cfg.train_discriminator,
        use_mbrl_demos=cfg.use_mbrl_demos,
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
        fixed_reward_value=0.0 if cfg.disc_binary_reward else None,
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
        fixed_reward_value=1.0 if cfg.disc_binary_reward else None,
    )
    if cfg.add_exp_to_replay_buffer:
        replay_buffer.add_batch(
            expert_dataset["observations"][:1000],
            expert_dataset["actions"][:1000],
            expert_dataset["next_observations"][:1000],
            expert_dataset["rewards"][:1000],
            expert_dataset["terminals"][:1000],
        )
    if cfg.from_end:
        print(
            f"{PrintColors.OKBLUE}Adding {cfg.overrides.expert_size} from end of expert dataset"
        )
        expert_replay_buffer.add_batch(
            expert_dataset["observations"][-cfg.overrides.expert_size :],
            expert_dataset["actions"][-cfg.overrides.expert_size :],
            expert_dataset["next_observations"][-cfg.overrides.expert_size :],
            expert_dataset["rewards"][-cfg.overrides.expert_size :],
            expert_dataset["terminals"][-cfg.overrides.expert_size :],
        )
    else:
        print(
            f"{PrintColors.OKBLUE}Adding {cfg.overrides.expert_size} from beginning of expert dataset"
        )
        expert_replay_buffer.add_batch(
            expert_dataset["observations"][: cfg.overrides.expert_size],
            expert_dataset["actions"][: cfg.overrides.expert_size],
            expert_dataset["next_observations"][: cfg.overrides.expert_size],
            expert_dataset["rewards"][: cfg.overrides.expert_size],
            expert_dataset["terminals"][: cfg.overrides.expert_size],
        )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------

    # --------------- SAC reset ratio schedule -----------------
    sac_reset_ratio = cfg.sac_schedule.start_ratio
    sac_reset_schedule = np.array(
        [
            [sac_reset_ratio, cfg.sac_schedule.mid_ratio, cfg.sac_schedule.m1],
            [
                cfg.sac_schedule.mid_ratio,
                cfg.sac_schedule.end_ratio,
                cfg.sac_schedule.m2,
            ],
        ]
    )
    sac_ratio_lag = 0
    if cfg.schedule_sac_ratio:
        print(f"{PrintColors.OKBLUE}Scheduling SAC reset ratio:")
        pp.pprint(sac_reset_schedule)
        print(PrintColors.ENDC)
    # -------------- discriminator lr schedule ------------------
    disc_lr = cfg.disc.start_lr
    disc_lr_schedule = np.array(
        [
            [disc_lr, cfg.disc.mid_lr, cfg.disc.m1],
            [cfg.disc.mid_lr, cfg.disc.end_lr, cfg.disc.m2],
        ]
    )
    disc_ratio_lag = 0
    if cfg.schedule_disc_special:
        print(f"{PrintColors.OKBLUE}Discriminator lr schedule:")
        pp.pprint(disc_lr_schedule)
        print(PrintColors.ENDC)
    if cfg.no_regret:
        print(f"{PrintColors.OKBLUE}No regret discriminator training")
        print(PrintColors.ENDC)
    else:
        print(f"{PrintColors.OKBLUE}Best response discriminator training")
        print(PrintColors.ENDC)
    drb = DiscriminatorReplayBuffer(obs_shape[0], act_shape[0])

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
        if cfg.disc_ensemble:
            f_net = DiscriminatorEnsemble(
                env, reduction=cfg.disc_ensemble_reduction
            ).to(cfg.device)
        else:
            f_net = Discriminator(env).to(cfg.device)
        f_opt = OAdam(f_net.parameters(), lr=disc_lr)
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

        if cfg.debug_mode and cfg.train_disc_in_model:
            # ! debug purposes only, remember to remove
            S_curr, A_curr, s = sample_from_learned_model(
                test_env,
                model_env,
                expert_sb3_agent,
                cfg.disc.num_traj_samples,
                rollout_length,
            )

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
            if (
                cfg.debug_mode
                or (env_steps + 1) % int(cfg.overrides.freq_train_model / 2) == 0
            ):
                # ! reset to 50/50 learner/expert states
                use_expert_data = rng.random() < cfg.overrides.model_exp_ratio
                model_train_buffer = replay_buffer
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    model_train_buffer,
                    work_dir=work_dir,
                )

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                # ! reset to expert states
                if cfg.schedule_sac_ratio:
                    (
                        sac_reset_schedule,
                        sac_reset_ratio,
                        sac_ratio_lag,
                    ) = mbrl.util.math.get_ratio(
                        sac_reset_schedule, env_steps, sac_ratio_lag
                    )
                reset_to_exp_states = rng.random() < sac_reset_ratio
                if cfg.use_yuda_default:
                    rollout_buffer = replay_buffer
                elif reset_to_exp_states:
                    rollout_buffer = expert_replay_buffer
                else:
                    rollout_buffer = policy_buffer
                rollout_model_and_populate_sac_buffer(
                    model_env,
                    rollout_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_batch_size,
                    fixed_reward_value=cfg.disc_binary_reward,
                )

                # ----------------------- Discriminator Training with Model ----------
                if cfg.debug_mode or (
                    cfg.update_with_model and cfg.train_discriminator
                ):
                    if cfg.schedule_disc_special:
                        (
                            disc_lr_schedule,
                            disc_lr,
                            disc_ratio_lag,
                        ) = mbrl.util.math.get_ratio(
                            disc_lr_schedule, env_steps, disc_ratio_lag
                        )
                    elif not disc_steps == 0:
                        disc_lr = cfg.disc.start_lr / disc_steps
                    else:
                        disc_lr = cfg.disc.start_lr
                    f_opt = OAdam(f_net.parameters(), lr=disc_lr)

                    if cfg.train_disc_in_model:
                        S_curr, A_curr, s = sample_from_learned_model(
                            test_env,
                            model_env,
                            agent,
                            cfg.disc.num_traj_samples,
                            rollout_length,
                        )
                    else:
                        S_curr, A_curr, s = sample(
                            test_env,
                            agent,
                            cfg.disc.num_traj_samples,
                            drb,
                            cfg.no_regret,
                        )
                    if cfg.no_regret and len(drb) > cfg.disc.batch_size:
                        S_curr, A_curr = drb.sample(cfg.disc.batch_size)
                    learner_sa_pairs = torch.cat(
                        (torch.from_numpy(S_curr), torch.from_numpy(A_curr)), dim=1
                    ).to(cfg.device)
                    # env_steps += s
                    # tbar.update(s)
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
                # ! which buffer is always sac_buffer because use_real_data is always False
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
                    # ! policy_exp_ratio == 0 for everything except pointmaze
                    # ! should update actor and critic on rollouts in the learned model
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
                            policy_buffer
                            if cfg.use_yuda_default or cfg.use_policy_buffer_adv_update
                            else sac_buffer,
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
                epoch += 1
            if (env_steps + 1) % cfg.eval_frequency == 0:
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
                            "sac_reset_ratio": sac_reset_ratio,
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
                    if cfg.train_discriminator:
                        print(f"{disc_lr=}")
                # if avg_reward > best_eval_reward:
                #     video_recorder.save(f"{epoch}.mp4")
                #     best_eval_reward = avg_reward
                #     agent.sac_agent.save_checkpoint(
                #         ckpt_path=os.path.join(work_dir, "sac.pth")
                #     )

                if cfg.train_disc_in_model:
                    # evaluate learner in learned model
                    pass

            tbar.update(1)
            env_steps += 1
            obs = next_obs

        # ------ Discriminator Training ------
        # if cfg.update_with_model:
        #     continue
        # if cfg.train_discriminator and updates_made % cfg.disc.freq_train_disc == 0:
        #     # print(f"Discriminator Training: {learning_rate_used}, {disc_steps}")
        #     if not disc_steps == 0:
        #         learning_rate_used = cfg.disc.lr / disc_steps
        #     else:
        #         learning_rate_used = cfg.disc.lr
        #     f_opt = OAdam(f_net.parameters(), lr=learning_rate_used)

        #     S_curr, A_curr, s = sample(test_env, agent, cfg.disc.num_traj_samples)
        #     learner_sa_pairs = torch.cat((S_curr, A_curr), dim=1).to(cfg.device)
        #     # env_steps += s    # * ignore env_steps for discriminator training
        #     tbar.update(s)
        #     for _ in range(cfg.disc.num_updates_per_step):
        #         learner_sa = learner_sa_pairs[
        #             np.random.choice(len(learner_sa_pairs), cfg.disc.batch_size)
        #         ]
        #         expert_batch = expert_replay_buffer.sample(cfg.disc.batch_size)
        #         expert_s, expert_a, *_ = cast(
        #             mbrl.types.TransitionBatch, expert_batch
        #         ).astuple()
        #         expert_sa = torch.cat((expert_s, expert_a), dim=1).to(cfg.device)
        #         f_opt.zero_grad()
        #         f_learner = f_net(learner_sa.float())
        #         f_expert = f_net(expert_sa.float())
        #         gp = gradient_penalty(learner_sa, expert_sa, f_net)
        #         loss = f_expert.mean() - f_learner.mean() + 10 * gp
        #         loss.backward()
        #         f_opt.step()
        #     disc_steps += 1

    return np.float32(best_eval_reward)
