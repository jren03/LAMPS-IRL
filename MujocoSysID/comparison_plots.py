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
    # return dict(alg=data.mean(axis=0))

    def iqm(x):
        return np.array([metrics.aggregate_iqm(x[:, :, i]) for i in range(x.shape[-1])])

    iqm_scores, _ = rly.get_interval_estimates(
        {"alg": np.expand_dims(data, 1)}, iqm, reps=5000
    )
    return iqm_scores


def main(env_abbrv, env_name, all_graphs):
    csv_results_dir = Path("csv_results", env_name)
    algs_to_colors = {
        "exp": "green",
        "mbpo": "grey",
        "sysid": "#4bacc6",
        "lamps": "#f0d38b",
    }
    algs_to_labels = {
        "exp": "Demo",
        "mbpo": "MBPO",
        "lamps": "LAMPS",
        "sysid": "SysID",
    }
    env_name_to_exp_scores = {
        # SB3 Experts
        "ant": 4596,
        "hc": 10317,
        "hop": 3399,
        "walk": 4395,
        "hum": 6160,
    }
    env_name_to_ptremble = {
        "ant": 0.01,
        "hc": 0.075,
        "hop": 0.01,
        "walk": 0.05,
        "hum": 0.025,
    }

    if all_graphs:
        model_based_algs = ["exp", "lamps"]
    else:
        model_based_algs = algs_to_colors.keys()

    # steps = 150
    steps = 250
    sz = 1000
    for alg in model_based_algs:
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
                shaky = "shaky" in csv.name
                if len(data) >= steps:
                    scores.append(data[:steps])
                else:
                    # extend last value to steps
                    print(f"Extending {alg} from {len(data)} to", end=" ")
                    scores.append(
                        np.concatenate([data, np.ones(steps - len(data)) * data[-1]])
                    )
                    print(f"{len(scores[-1])}")
            if scores == []:
                print(f"Skipping {alg}")
                continue
            scores = np.stack(scores, axis=0)
            mean, std_err = (
                calc_iqm(scores)["alg"],
                np.std(scores, axis=0) / np.sqrt(len(scores)),
            )
            plt.plot(
                np.arange(steps) * sz,
                mean,
                color=algs_to_colors[alg],
                label=f"{algs_to_labels[alg]}: {len(scores)} runs",
            )
            plt.fill_between(
                np.arange(steps) * sz,
                mean - std_err,
                mean + std_err,
                color=algs_to_colors[alg],
                alpha=0.1,
            )
            print(f"Plotting {alg} with {len(scores)} runs")

    if shaky:
        p_tremble = env_name_to_ptremble[env_abbrv]
    else:
        p_tremble = 0
    env_name = env_name.replace("gym___", "")

    if all_graphs:
        if env_name == "ant_truncated_obs":
            env_name = "Ant-v3"
        elif env_name == "humanoid_truncated_obs":
            env_name = "Humanoid-v3"
        model_based_sz, model_based_steps = sz, steps
        sz = 10000 * 5
        steps = (model_based_sz * model_based_steps) // sz + 1
        model_free_root = Path("/share/portal/jlr429/pessimistic-irl/fast_irl")
        bc = []
        for s in range(11):
            path = Path(
                model_free_root, f"learners/{env_name}/bc_{s}_pt{p_tremble}.npz"
            )
            if not path.exists():
                continue
            bc.append(
                [np.load(x, allow_pickle=True)["means"] for x in glob.glob(str(path))][
                    0
                ]
            )
        if bc:
            bc = np.stack(bc, axis=0).reshape(-1, 1)
            bc_mean, bc_ci = (
                calc_iqm(bc)["alg"],
                np.std(bc, axis=0) / np.sqrt(len(bc)),
            )
            plt.plot(
                np.arange(steps) * sz,
                np.ones(steps) * bc_mean,
                color="#8064A2",
                linestyle="--",
                label="BC",
            )
            plt.fill_between(
                np.arange(steps) * sz,
                bc_mean + bc_ci,
                bc_mean - bc_ci,
                color="#8064A2",
                alpha=0.1,
            )
        # mm
        mm = []
        for s in range(1, 11):
            path = Path(
                model_free_root,
                f"results/10seeds/{env_name}/mm_icml_s{s}_alpha_0.0_relabel_pt{p_tremble}.npz",
            )
            if not path.exists():
                continue
            for x in glob.glob(str(path)):
                means = np.load(x, allow_pickle=True)["means"]
                if len(means) < steps:
                    continue
                else:
                    mm.append(means[:steps])
        if mm:
            mm = np.stack(mm, axis=0)
            mm_mean, mm_ci = (
                calc_iqm(mm)["alg"],
                np.std(mm, axis=0) / np.sqrt(len(mm)),
            )
            plt.plot(np.arange(steps) * sz, mm_mean, label="MM", color="grey")
            plt.fill_between(
                np.arange(steps) * sz,
                mm_mean - mm_ci,
                mm_mean + mm_ci,
                color="grey",
                alpha=0.1,
            )
            print(f"Plotting MM with {len(mm)} runs")

        # # mm-bal
        mm_bal = []
        for s in range(1, 11):
            path = Path(
                model_free_root,
                f"results/10seeds/{env_name}/mm_icml_s{s}_alpha_0.0_relabel_schedule_pt{p_tremble}.npz",
            )
            if not path.exists():
                continue
            for x in glob.glob(str(path)):
                means = np.load(x, allow_pickle=True)["means"]
                if len(means) < steps:
                    continue
                else:
                    mm_bal.append(means[:steps])
        if mm_bal:
            mm_bal = np.stack(mm_bal, axis=0)
            mm_bal_mean, mm_bal_ci = (
                calc_iqm(mm_bal)["alg"],
                np.std(mm_bal, axis=0) / np.sqrt(len(mm_bal)),
            )
            plt.plot(np.arange(steps) * sz, mm_bal_mean, label="HyPE", color="#F79646")
            plt.fill_between(
                np.arange(steps) * sz,
                mm_bal_mean - mm_bal_ci,
                mm_bal_mean + mm_bal_ci,
                color="#F79646",
                alpha=0.1,
            )
            print(f"Plotting HyPE with {len(mm_bal)} runs")

        # filter-br alpha = 0.5
        filt_br_a05 = []
        for s in range(1, 11):
            path = Path(
                model_free_root,
                f"results/10seeds/{env_name}/filter_br_icml_s{s}_alpha_0.5_relabel_pt{p_tremble}.npz",
            )
            if not path.exists():
                continue
            for x in glob.glob(str(path)):
                means = np.load(x, allow_pickle=True)["means"]
                if len(means) < steps:
                    continue
                else:
                    filt_br_a05.append(means[:steps])
        if filt_br_a05:
            filt_br_a05 = np.stack(filt_br_a05, axis=0)
            br_05_mean, br_05_ci = (
                calc_iqm(filt_br_a05)["alg"],
                np.std(filt_br_a05, axis=0) / np.sqrt(len(filt_br_a05)),
            )
            plt.plot(np.arange(steps) * sz, br_05_mean, label="FILTER", color="#4bacc6")
            plt.fill_between(
                np.arange(steps) * sz,
                br_05_mean - br_05_ci,
                br_05_mean + br_05_ci,
                color="#4bacc6",
                alpha=0.1,
            )
            print(f"Plotting FILTER with {len(filt_br_a05)} runs")

        # iq_learn
        iq_learn = []
        for s in range(11):
            path = Path(
                model_free_root,
                f"/share/portal/jlr429/pessimistic-irl/official_iqlearn/iq_learn/npz_results/{env_name}_100000_sb3_1000000_steps/iq_learn_seed_{s}_pt{p_tremble}.npz",
            )
            if not path.exists():
                continue
            for x in glob.glob(str(path)):
                means = np.load(x, allow_pickle=True)["means"]
                if len(means) < steps:
                    continue
                else:
                    iq_learn.append(means[:steps])
        if iq_learn:
            iq_learn = np.stack(iq_learn, axis=0)
            iq_mean, iq_ci = (
                calc_iqm(iq_learn)["alg"],
                np.std(iq_learn, axis=0) / np.sqrt(len(iq_learn)),
            )
            plt.plot(np.arange(steps) * sz, iq_mean, label="IQ-Learn", color="#F497DA")
            plt.fill_between(
                np.arange(steps) * sz,
                iq_mean - iq_ci,
                iq_mean + iq_ci,
                color="#F497DA",
                alpha=0.1,
            )
            print(f"Plotting IQ-Learn with {len(iq_learn)} runs")

    plt.legend(ncol=2, fontsize=8)
    plt.ylabel("IQM of $J(\\pi)$")
    plt.xlabel("Env. Steps")
    plt.xticks(rotation=45)
    plt.title(f"{env_name}, " + "$p_{tremble}=$" + str(p_tremble))
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
    parser.add_argument("-a", "--all_graphs", action="store_true", default=False)
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
    if "truncated" not in env_name:
        env_name = f"gym___{env_name}"
    main(env_abbr, env_name, args.all_graphs)
