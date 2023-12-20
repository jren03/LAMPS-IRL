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


def main(env_abbrv, env_name):
    csv_results_dir = Path("csv_results", env_name)
    algs_to_colors = {
        "exp": "green",
        "sysid": "grey",
        "lamps": "#F79646",
    }
    algs_to_labels = {
        "exp": "Demo",
        "lamps": "LAMPS",
        "sysid": "SysID",
    }
    env_name_to_exp_scores = {
        "ant": 5207.53,
        "hc": 13088.32,
        "hop": 3318.71,
        "hum": 5708.24,
        "walk": 5740.80,
    }

    steps = 150
    sz = 2000
    for alg in algs_to_colors.keys():
        if alg == "exp":
            plt.plot(
                np.arange(steps) * sz,
                np.ones(steps) * env_name_to_exp_scores[env_abbrv],
                color=algs_to_colors[alg],
                linestyle="--",
                label=algs_to_labels[alg],
            )
        else:
            csvs = [f for f in csv_results_dir.glob("*.csv") if alg in f.name]
            scores = []
            for csv in csvs:
                data = pd.read_csv(csv).episode_reward.to_numpy()
                scores.append(data)
            scores = np.stack(scores, axis=0)
            mean, std_err = (
                calc_iqm(scores)["alg"],
                np.std(scores, axis=0) / np.sqrt(len(scores)),
            )
            plt.plot(
                np.arange(steps) * sz,
                mean,
                color=algs_to_colors[alg],
                label=algs_to_labels[alg],
            )
            plt.fill_between(
                np.arange(steps) * sz,
                mean - std_err,
                mean + std_err,
                color=algs_to_colors[alg],
                alpha=0.1,
            )
            print(f"Plotting {alg} with {len(scores)} runs")

    env_name = env_name.replace("gym___", "")
    plt.legend(ncol=2, fontsize=8)
    plt.ylabel("IQM of $J(\\pi)$")
    plt.xlabel("Env. Steps")
    plt.xticks(rotation=45)
    plt.title(f"{env_name}, " + "$p_{tremble}=$" + str(0))
    plt.savefig(f"plots/{env_name}.png", bbox_inches="tight")
    print("SAVED")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        choices=["ant", "hc", "hop", "hum", "walk"],
        help="Name of the environment",
    )
    parser.add_argument("-o", "--override", action="store_true", default=False)
    args = parser.parse_args()

    env_abbr = args.env_name
    if env_abbr == "ant":
        env_name = "Ant-v3"
    elif env_abbr == "hc":
        env_name = "HalfCheetah-v3"
    elif env_abbr == "hop":
        env_name = "Hopper-v3"
    elif env_abbr == "hum":
        env_name = "Humanoid-v3"
    elif env_abbr == "walk":
        env_name = "Walker2d-v3"
    env_name = f"gym___{env_name}"
    main(env_abbr, env_name)
