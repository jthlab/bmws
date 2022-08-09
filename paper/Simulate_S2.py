#Performance if we incorrectly assume that Ne is constant

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

fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(12, 9))

axs[0][0].axis("off")
optimal_lambda = [5, 4, 3]

for i in range(1, 3):
    axs[i][0].plot(range(1, 101), Ne_mdls[i], color="black")

for j in range(3):
    axs[0][j + 1].plot(range(1, 101), s_mdls[j]["s"], color="black")

for i in range(3):
    for j in range(1, 3):
        res = []
        for seed in range(N_SIMULATIONS):
            res.append(
                sim_and_fit(
                    s_mdls[i],
                    seed=12345 + seed,
                    lam=10 ** optimal_lambda[i],
                    Ne=Ne_mdls[j],
                    Ne_fit=[10000] * len(s_mdls[i]["s"]),
                    em_iterations=N_EM_ITERATIONS,
                )
            )
        x, y = zip(*[(range(len(rr["s_hat"])), rr["s_hat"]) for rr in res])
        plot_summary(axs[j][i + 1], x, y, s_mdls[i]["s"])

fig.savefig("plots/Simulate2.pdf")
