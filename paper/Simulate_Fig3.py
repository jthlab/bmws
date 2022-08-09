#Smoothing parameter

from IPython.display import set_matplotlib_formats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import jax
from math import log, exp, sqrt
from sim import sim_and_fit, sim_wf

# from estimate import posterior_decoding, sample_paths
from common import Observation
from plotting import plot_summary, compute_rmse, bias_variance
from itertools import combinations_with_replacement, product

rng = np.random.default_rng()

N_SIMULATIONS = 100
N_EM_ITERATIONS = 3
set_matplotlib_formats("svg")

s_mdls = [
    {"s": [0.01] * 100, "h": [0.5] * 100, "f0": 0.1},
    #    {"s": [0.02] * 50 + [-0.02] * 50, "h": [0.5] * 100, "f0": 0.1},
    {
        "s": [0.02 * np.cos(np.pi * x / 99) for x in range(100)],
        "h": [0.5] * 100,
        "f0": 0.1,
    },
    {"s": (([0.02] * 20 + [-0.02] * 20) * 3)[:100], "h": [0.5] * 100, "f0": 0.5},
]
s_names = ["const", "switch", "fluc"]

gr = exp(log(10) / 100)
Ne_mdls = [
    [10000] * 100,  # constant
    [round(10000 * gr ** (10 * int(x / 10))) for x in range(100)],  # exp growth
    [10000] * 40 + [1000] * 20 + [10000] * 40,  # bottleneck
]
Ne_names = ["const", "exp", "bottle"]

###########################################################################

samples = pd.read_csv("data/allbrit.meta", sep="\t")
samples["GenBP"] = samples.DateBP // 29  # assume 29 years per-generation
counts = samples.GenBP.value_counts().sort_index()
sizes, times = counts.values, np.array(counts.index)

T = times[-1]
s2_mdls = [
    {"s": [0.01] * T, "h": [0.5] * T, "f0": 0.1},
    {
        "s": [-0.02 * np.cos(np.pi * x / (T - 1)) for x in range(T)],
        "h": [0.5] * T,
        "f0": 0.1,
    },
    {"s": (([0.02] * 20 + [-0.02] * 20) * 6)[:T], "h": [0.5] * T, "f0": 0.5},
]
gr = exp(log(100) / T)
Ne_mdl2 = [round(10000 * gr ** (10 * int(x / 10))) for x in range(T)]  # exp growth

fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 6))

optimal_lambda = [5, 4, 2]


for j in range(3):
    res = []
    for seed in range(N_SIMULATIONS*10):
        this_res = sim_and_fit(s2_mdls[j], seed=12345 + seed, lam=10**optimal_lambda[j], Ne=10000, k=times, n=sizes)
        res.append(this_res)
    x, y = zip(*[(range(len(rr["s_hat"])), rr["s_hat"]) for rr in res])
    plot_summary(axs[0][j], x, y, s2_mdls[j]["s"])
    axs[0][j].invert_xaxis()
    axs[0][j].set_xlabel("Generations before present")

axs[0][0].set_ylabel("Selection coefficient")

log10_lambda = [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6]

for j in range(3):
    plot_res = []  # results to make this subplot
    for lam in log10_lambda:
        this_res = []
        for seed in range(N_SIMULATIONS):
            this_res.append(
                sim_and_fit(s2_mdls[j], seed=12345 + seed, lam=10 ** lam, Ne=10000, k=times, n=sizes,em_iterations=N_EM_ITERATIONS)
            )

        x, y = zip(*[(range(len(rr["s_hat"])), rr["s_hat"]) for rr in this_res])
        b, v, m = bias_variance(y, s2_mdls[j]["s"])
        plot_res.append(
            {
                "s_mdl": s_names[j],
                "Ne_mdl": "Exp2",
                "lambda": lam,
                "rmse": m,
                "rbias": b,
                "rvar": v,
            }
        )


    plot_res = pd.DataFrame(plot_res)
    #axs[1][j].tick_params(labelsize=6)
    axs[1][j].xaxis.label.set_visible(False)
    axs[1][j].title.set_visible(False)
    axs[1][j].plot(plot_res["lambda"], plot_res["rmse"], color="tab:red")
    axs[1][j].plot(plot_res["lambda"], plot_res["rbias"], color="tab:green")
    axs[1][j].plot(plot_res["lambda"], plot_res["rvar"], color="tab:blue")
    axs[1][j].set_xlabel("Smoothing parameter λ")

axs[1][0].set_ylabel("Root mean squared error")

fig.suptitle(None)

fig.savefig("plots/Simulate5.pdf")
