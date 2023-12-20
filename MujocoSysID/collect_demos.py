from pathlib import Path
import gym
import torch
import h5py
import hydra
import omegaconf
import numpy as np
from tqdm import tqdm, trange

from typing import cast
import mbrl.third_party.pytorch_sac_pranz24 as pytorch_sac_pranz24
import mbrl.util.env
from mbrl.planning.sac_wrapper import SACAgent
from mbrl.planning import complete_agent_cfg

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_reset_data():
    data = dict(
        observations=[],
        next_observations=[],
        actions=[],
        rewards=[],
        terminals=[],
        timeouts=[],
        logprobs=[],
        qpos=[],
        qvel=[],
    )
    return data


def rollout(policy, cfg, max_path=1000, num_data=100_000):
    env, _, _ = mbrl.util.env.EnvHandler.make_env(cfg)

    data = get_reset_data()
    traj_data = get_reset_data()

    total_rew = []
    _returns = 0
    t = 0
    done = False
    s = env.reset()
    while len(data["rewards"]) < num_data:
        # torch_s = torch.from_numpy(np.expand_dims(s, axis=0)).float().to(device)
        # a = policy.select_action(s)
        a = policy.act(s)
        # a = a.cpu().detach().numpy().squeeze()

        # mujoco only
        qpos, qvel = env.sim.data.qpos.ravel().copy(), env.sim.data.qvel.ravel().copy()

        try:
            ns, rew, done, infos = env.step(a)
        except:
            print("lost connection")
            env.close()
            env, _, _ = mbrl.util.env.EnvHandler.make_env(cfg)
            s = env.reset()
            policy.reset()
            traj_data = get_reset_data()
            t = 0
            _returns = 0
            continue

        _returns += rew

        t += 1
        timeout = False
        terminal = False
        if t == max_path:
            timeout = True
        elif done:
            terminal = True

        traj_data["observations"].append(s)
        traj_data["actions"].append(a)
        traj_data["next_observations"].append(ns)
        traj_data["rewards"].append(rew)
        traj_data["terminals"].append(terminal)
        traj_data["timeouts"].append(timeout)
        traj_data["qpos"].append(qpos)
        traj_data["qvel"].append(qvel)

        s = ns
        if terminal or timeout:
            print(
                "Finished trajectory. Len=%d, Returns=%f. Progress:%d/%d"
                % (t, _returns, len(data["rewards"]), num_data)
            )
            total_rew.append(_returns)
            s = env.reset()
            t = 0
            _returns = 0
            for k in data:
                data[k].extend(traj_data[k])
            traj_data = get_reset_data()

    new_data = dict(
        observations=np.array(data["observations"]).astype(np.float32),
        actions=np.array(data["actions"]).astype(np.float32),
        next_observations=np.array(data["next_observations"]).astype(np.float32),
        rewards=np.array(data["rewards"]).astype(np.float32),
        terminals=np.array(data["terminals"]).astype(bool),
        timeouts=np.array(data["timeouts"]).astype(bool),
        qpos=np.array(data["qpos"]).astype(np.float32),
        qvel=np.array(data["qvel"]).astype(np.float32),
    )
    print(f"{np.mean(total_rew)=}, {np.std(total_rew)=}")

    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    return new_data


@hydra.main(
    config_path="/share/portal/jlr429/pessimistic-irl/LAMPS-IRL/MujocoSysID/mbrl/examples/conf",
    config_name="main",
)
def collect_demos(cfg: omegaconf.DictConfig):
    env_name = cfg.overrides.env.replace("gym___", "")

    num_data = 100_000
    save_dir = Path("/share/portal/jlr429/pessimistic-irl/expert_data")
    save_dir.mkdir(exist_ok=True)
    save_path = Path(save_dir, f"{env_name}_{num_data}_mbrl.h5")

    env, _, _ = mbrl.util.env.EnvHandler.make_env(cfg)
    complete_agent_cfg(env, cfg.algorithm.agent)
    expert = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(cfg.algorithm.agent)),
    )
    expert.sac_agent.load_checkpoint(
        ckpt_path=Path(
            f"/share/portal/jlr429/pessimistic-irl/LAMPS-IRL/MujocoSysID/expert/{env_name}/sac.pth"
        ),
        evaluate=True,
    )

    data = rollout(
        expert,
        cfg,
        max_path=1000,
        num_data=100_000,
    )

    hfile = h5py.File(save_path, "w")
    for k in data:
        hfile.create_dataset(k, data=data[k], compression="gzip")
    hfile.close()


if __name__ == "__main__":
    collect_demos()
