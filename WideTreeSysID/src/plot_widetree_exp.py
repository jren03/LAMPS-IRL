import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['grid.linewidth'] = 0.5
plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = (7,6)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

our = "\\textsc{LAMPS-MM}"
sysid = "\\textsc{Sysid}"

def plot_data(seeds):
    mle_weights = []
    moment_based_weights = []

    for seed in seeds:
        mle_weights.append(np.load(f"data/mle_{seed}.npy"))
        moment_based_weights.append(np.load(f"data/moment_based_{seed}.npy"))

    mle_mean_weights, mle_std_weights = np.mean(mle_weights, axis=0), np.std(mle_weights, axis=0)
    moment_based_mean_weights, moment_based_std_weights = np.mean(moment_based_weights, axis=0), np.std(moment_based_weights, axis=0)

    plt.clf()
    xrange = np.arange(mle_mean_weights.shape[0]) * 3

    color = CB_color_cycle[0]
    for model_idx in range(moment_based_mean_weights.shape[1]):
        plt.plot(xrange, moment_based_mean_weights[:, model_idx], color=color, label=our + " $M^{\mathsf{good}}$" if model_idx > 0 else our + " $M^{\mathsf{bad}}$", linestyle="dashed" if model_idx ==0 else "solid")
        plt.fill_between(xrange, moment_based_mean_weights[:, model_idx] - moment_based_std_weights[:, model_idx], moment_based_mean_weights[:, model_idx] + moment_based_std_weights[:, model_idx], color=color, alpha=0.2)

    color = CB_color_cycle[1]
    for model_idx in range(mle_mean_weights.shape[1]):
        plt.plot(xrange, mle_mean_weights[:, model_idx], color=color, label=sysid+" $M^{\mathsf{good}}$" if model_idx > 0 else sysid+" $M^{\mathsf{bad}}$", linestyle="dashed" if model_idx ==0 else "solid")
        plt.fill_between(xrange, mle_mean_weights[:, model_idx] - mle_std_weights[:, model_idx], mle_mean_weights[:, model_idx] + mle_std_weights[:, model_idx], color=color, alpha=0.2)

    # plt.xlabel("Number of Iterations")
    # plt.ylabel("Probability using classification loss")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("mle_widetree_exp.png")

    # plt.clf()    

    plt.xlabel("Number of real world interactions\n(d)")
    plt.ylabel("Probability of picking model")
    plt.legend()
    plt.grid(True)
    plt.title("widetree")
    # plt.savefig("moment_based_widetree_exp.png")
    plt.savefig("widetree_exp.pdf", bbox_inches='tight')


if __name__ == "__main__":
    seeds = range(10)
    plot_data(seeds)
    