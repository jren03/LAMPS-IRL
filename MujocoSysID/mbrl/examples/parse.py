from mbrl.util.psdp_fetch_demos import fetch_demos
from pathlib import Path
import numpy as np

env_names = ["antmaze-large-diverse-v2", "antmaze-large-play-v2"]

for env_name in env_names:
    print(f"{env_name=}")
    (
        new_dataset,
        expert_sa_pairs,
        qpos,
        qvel,
        goals,
        expert_reset_states,
    ) = fetch_demos(env_name)

    # save to npz file
    expert_data_root = Path("/share/portal/jlr429/pessimistic-irl/expert_data")
    np.savez(
        # f"{env_name}_demos.npz",
        str(Path(expert_data_root, f"{env_name}_psdp_demos.npz")),
        observations=new_dataset["observations"],
        actions=new_dataset["actions"],
        next_observations=new_dataset["next_observations"],
        rewards=new_dataset["rewards"],
        terminals=new_dataset["terminals"],
        expert_sa_pairs=expert_sa_pairs.numpy(),
        expert_reset_states=expert_reset_states,
        qpos=qpos,
        qvel=qvel,
        goals=goals,
    )
    print(f"Saved to {expert_data_root}/{env_name}_psdp_demos.npz")
