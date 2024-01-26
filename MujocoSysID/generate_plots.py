from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib
import scipy.stats
import matplotlib.gridspec as gridspec
import pdb
import glob
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from collections import defaultdict

sns.set(font="serif", font_scale=1.4)
sns.set_style(
    "white",
    {
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"],
        "axes.facecolor": "white",
        "lines.markeredgewidth": 1,
    },
)


def setup_plot():
    fig = plt.figure(dpi=100, figsize=(7.0, 6.0))
    ax = plt.subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=15)
    ax.tick_params(direction="in")


def calc_iqm(data):
    iqm = lambda x: np.array(
        [metrics.aggregate_iqm(x[:, :, i]) for i in range(x.shape[-1])]
    )
    iqm_scores, _ = rly.get_interval_estimates(
        {"alg": np.expand_dims(data, 1)}, iqm, reps=5000
    )
    return iqm_scores


def main(env_abbrv, env_name, steps=15):
    csv_results_dir = Path("csv_results", env_name)
    algs_to_colors = {
        "exp": "green",
        "mbpo": "grey",
        "sysid": "#4bacc6",
        "lamps": "#F79646",
        "bc": "#8064A2",
    }
    algs_to_labels = {
        "exp": "Demo",
        "mbpo": "MBPO",
        "lamps": "LAMPS",
        "sysid": "SysID",
        "bc": "BC",
    }
    env_name_to_exp_scores = {
        # SB3 Experts
        "ant": 4596,
        "hc": 10317,
        "hop": 3399,
        "walk": 4395,
        "hum": 6160,
        ### Yuda Experts
        # "ant": 5207.53,
        # "hc": 13088.32,
        # "hop": 3318.71,
        # "hum": 5708.24,
        # "walk": 5740.80,
    }
    env_name_to_bc_scores = {
        "div": 30.0,
        "play": 35.2,
    }
    env_name_to_ptremble = {
        "ant": 0.01,
        "hc": 0.075,
        "hop": 0.01,
        "walk": 0.05,
        "hum": 0.025,
        "div": 0.0,
        "play": 0.0,
    }

    # steps = 150
    # steps = 100
    # sz = 1000

    reset_versions = [
        "psdp",
        "nrpi",
        "backward-sw",
        "forward-sw",
        "backward-range",
        "forward-range",
    ]
    # create a subplot for each reset version, and a map from
    # reset version to the subplot
    reset_version_to_ax = {}
    reset_version_to_scores = defaultdict(list)
    fig = plt.figure(dpi=100, figsize=(7.0, 6.0))
    gs = gridspec.GridSpec(2, 3)
    for i, reset_version in enumerate(reset_versions):
        ax = plt.subplot(gs[i])
        reset_version_to_ax[reset_version] = ax

    # steps = 10
    sz = 10_000
    partition = ""
    reset_version_to_steps = defaultdict(int)
    for alg in algs_to_colors.keys():
        if alg == "exp" and env_abbrv in env_name_to_exp_scores.keys():
            plt.plot(
                np.arange(steps) * sz,
                np.ones(steps) * env_name_to_exp_scores[env_abbrv],
                color=algs_to_colors[alg],
                linestyle="--",
                label=algs_to_labels[alg],
            )
        elif alg == "bc" and env_abbrv in env_name_to_bc_scores.keys():
            # plot BC on each subplot
            for reset_version, ax in reset_version_to_ax.items():
                steps = max(reset_version_to_steps[reset_version], 2)
                ax.plot(
                    np.arange(steps) * sz,
                    np.ones(steps) * env_name_to_bc_scores[env_abbrv],
                    color=algs_to_colors[alg],
                    linestyle="--",
                    label=algs_to_labels[alg],
                )
        else:
            csvs = [f for f in csv_results_dir.glob("*.csv") if alg in f.name]
            # scores = []
            appended = False
            for csv in csvs:
                reset_version = csv.stem.split("_")[-2]
                data = pd.read_csv(csv).episode_reward.to_numpy()
                shaky = "shaky" in csv.name
                print(
                    f"Read {csv} with {len(data)} entries: reset_version={reset_version}"
                )
                reset_version_to_steps[reset_version] = max(
                    reset_version_to_steps[reset_version], len(data)
                )

            for csv in csvs:
                reset_version = csv.stem.split("_")[-2]
                scores = reset_version_to_scores[reset_version]
                steps = reset_version_to_steps[reset_version]
                if steps == 1:
                    steps += 1
                data = pd.read_csv(csv).episode_reward.to_numpy()
                if len(data) >= steps:
                    scores.append(data[:steps])
                    appended = True
                elif len(data) > steps - 3:
                    # extend last value to steps
                    print(f"Extending {alg} from {len(data)} to", end=" ")
                    scores.append(
                        np.concatenate([data, np.ones(steps - len(data)) * data[-1]])
                    )
                    print(f"{len(scores[-1])}")
                    appended = True
                partition = csv.stem.split("_")[-1]

            if not appended:
                print(f"Skipping {alg}")
                continue

            # loop through each reset verion and plot
            for reset_version, ax in reset_version_to_ax.items():
                steps = reset_version_to_steps[reset_version]
                if steps == 1:
                    steps += 1
                scores = reset_version_to_scores[reset_version]
                if scores == []:
                    continue
                scores = np.stack(scores, axis=0)
                mean, std_err = (
                    calc_iqm(scores)["alg"],
                    np.std(scores, axis=0) / np.sqrt(len(scores)),
                )
                ax.plot(
                    np.arange(steps) * sz,
                    mean,
                    color=algs_to_colors[alg],
                    label=f"{algs_to_labels[alg]}: {len(scores)} runs",
                )
                ax.fill_between(
                    np.arange(steps) * sz,
                    mean - std_err,
                    mean + std_err,
                    color=algs_to_colors[alg],
                    alpha=0.1,
                )
                print(f"Plotting {alg} with {len(scores)} runs")
                ax.set_title(reset_version)
            # scores = np.stack(scores, axis=0)
            # mean, std_err = (
            #     calc_iqm(scores)["alg"],
            #     np.std(scores, axis=0) / np.sqrt(len(scores)),
            # )
            # plt.plot(
            #     np.arange(steps) * sz,
            #     mean,
            #     color=algs_to_colors[alg],
            #     label=f"{algs_to_labels[alg]}: {len(scores)} runs",
            # )
            # plt.fill_between(
            #     np.arange(steps) * sz,
            #     mean - std_err,
            #     mean + std_err,
            #     color=algs_to_colors[alg],
            #     alpha=0.1,
            # )
            # print(f"Plotting {alg} with {len(scores)} runs")

    if shaky:
        p_tremble = env_name_to_ptremble[env_abbrv]
    else:
        p_tremble = 0

    env_name = env_name.replace("gym___", "")
    plt.legend(ncol=2, fontsize=8)
    plt.ylabel("IQM of $J(\\pi)$")
    plt.xlabel("Env. Steps")
    plt.xticks(rotation=45)
    plt.suptitle(f"{env_name}, " + "$p_{tremble}=$" + str(p_tremble) + ", " + partition)
    plt.tight_layout()
    plt.savefig(f"plots/{env_name}.png", bbox_inches="tight")
    print("SAVED")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        choices=["ant", "hc", "hop", "hum", "walk", "div", "play"],
        help="Name of the environment",
    )
    parser.add_argument("-s", "--steps", type=int, help="Number of steps", default=15)
    args = parser.parse_args()

    env_abbr = args.env_name
    if env_abbr == "ant":
        env_name = "ant_truncated_obs"
    elif env_abbr == "hc":
        env_name = "HalfCheetah-v3"
    elif env_abbr == "hop":
        env_name = "Hopper-v3"
    elif env_abbr == "hum":
        env_name = "humanoid_truncated_obs"
    elif env_abbr == "walk":
        env_name = "Walker2d-v3"
    elif env_abbr == "play":
        env_name = "antmaze-large-play-v2"
    elif env_abbr == "div":
        env_name = "antmaze-large-diverse-v2"
    if "truncated" not in env_name:
        env_name = f"gym___{env_name}"
    main(env_abbr, env_name, steps=args.steps)
