# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from tabnanny import verbose
from typing import Optional, Sequence, cast
import time

import gym
import hydra.utils
import numpy as np
import omegaconf
import torch
import pprint

from stable_baselines3.common.evaluation import evaluate_policy

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math
from mbrl.env.gym_wrappers import ResetWrapper
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.third_party.pytorch_sac import VideoRecorder
from mbrl.util.fetch_demos import fetch_demos
from mbrl.models.discriminator import Discriminator, DiscriminatorEnsemble
from mbrl.util.oadam import OAdam
from mbrl.util.common import gradient_penalty, PrintColors
from mbrl.util.discriminator_replay_buffer import DiscriminatorReplayBuffer
from mbrl.util.hybrid_replay_buffer import HybridReplayBuffer
from mbrl.util.sac_relabel_rewards import SACRelabelRewards
from typing import Callable, Union

import d4rl
from tqdm import tqdm

import stable_baselines3 as sb3
from pathlib import Path
import warnings


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
            action = agent.predict(obs)[0]
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


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


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

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    is_maze = "maze" in cfg.overrides.env
    expert_dataset, qpos, qvel = fetch_demos(
        cfg.overrides.env,
        zero_out_rewards=cfg.train_discriminator,
        use_mbrl_demos=cfg.use_mbrl_demos,
    )

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
    # random_explore = cfg.algorithm.random_initial_explore
    # mbrl.util.common.rollout_agent_trajectories(
    #     env,
    #     cfg.algorithm.initial_exploration_steps,
    #     mbrl.planning.RandomAgent(env) if random_explore else agent,
    #     {} if random_explore else {"sample": True, "batched": False},
    #     replay_buffer=replay_buffer,
    #     additional_buffer=policy_buffer,
    # )

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
                env, n_discriminators=cfg.n_discs, reduction=cfg.disc_ensemble_reduction
            ).to(cfg.device)
        else:
            f_net = Discriminator(env).to(cfg.device)
        f_opt = OAdam(f_net.parameters(), lr=disc_lr)
        model_env = mbrl.models.ModelEnv(
            env, dynamics_model, termination_fn, f_net, generator=torch_generator
        )
        # agent.sac_agent.add_f_net(f_net)
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

    if not cfg.hyirl:
        env = ResetWrapper(env, qpos=qpos, qvel=qvel)
    agent = SACRelabelRewards(
        f_net,
        "MlpPolicy",
        env,
        verbose=0,
        policy_kwargs=dict(net_arch=[256, 256]),
        ent_coef="auto",
        learning_rate=linear_schedule(7.3e-4),
        train_freq=64,
        gradient_steps=64,
        gamma=0.98,
        tau=0.02,
        device=cfg.device,
    )
    agent.actor.optimizer = OAdam(agent.actor.parameters())
    agent.critic.optimizer = OAdam(agent.critic.parameters())
    agent.replay_buffer = HybridReplayBuffer(
        agent.buffer_size,
        agent.observation_space,
        agent.action_space,
        agent.device,
        1,
        agent.optimize_memory_usage,
        expert_data=expert_dataset,
        balanced_sampling=cfg.hyirl,
        fixed_hybrid_schedule=False,
    )

    best_eval_reward = -np.inf
    epoch = 0
    disc_steps = 0
    sac_buffer = None

    tbar = tqdm(range(cfg.overrides.num_steps), ncols=0)
    while env_steps < cfg.overrides.num_steps:
        agent.learn(total_timesteps=10_000)
        tbar.update(10_000)
        env_steps += 10_000

        # ------ Discriminator Training ------
        if (
            cfg.train_discriminator
            and not cfg.update_with_model
            and updates_made != 0
            and (updates_made) % cfg.disc.freq_train_disc == 0
        ):
            if not disc_steps == 0:
                disc_lr = cfg.disc.start_lr / disc_steps
            else:
                disc_lr = cfg.disc.start_lr
            f_opt = OAdam(f_net.parameters(), lr=disc_lr)
            S_curr, A_curr, s = sample(
                test_env,
                agent,
                cfg.disc.num_traj_samples,
                drb,
                cfg.no_regret,
            )
            learner_sa_pairs = torch.cat(
                (torch.from_numpy(S_curr), torch.from_numpy(A_curr)), dim=1
            ).to(cfg.device)
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

        # ------ Epoch ended (evaluate and save model) ------
        if (env_steps + 1) % cfg.overrides.epoch_length == 0:
            epoch += 1
        if (env_steps + 1) % cfg.eval_frequency == 0:
            if not is_maze:
                # start_time = time.time()
                mean_reward, std_reward = evaluate_policy(
                    agent, test_env, n_eval_episodes=2
                )
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "episode_reward": mean_reward,
                        "sac_reset_ratio": sac_reset_ratio,
                    },
                )
                # print(f"Time for evaluation: {time.time() - start_time}")
            else:
                mean_reward, std_reward = evaluate_policy(
                    agent, test_env, n_eval_episodes=25
                )
                logger.log_data(
                    mbrl.constants.RESULTS_LOG_NAME,
                    {
                        "epoch": epoch,
                        "env_step": env_steps,
                        "episode_reward": mean_reward,
                        "success_rate": 1,
                    },
                )

    return np.float32(best_eval_reward)
