# Performance under standard models

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
    [100000] * 100,  # constant
    [round(100000 * gr ** (10 * int(x / 10))) for x in range(100)],  # exp growth
    [100000] * 40 + [1000] * 20 + [10000] * 40,  # bottleneck
]
Ne_names = ["const", "exp", "bottle"]

###########################################################################

fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(12, 12))

axs[0][0].axis("off")

optimal_lambda = [5, 4, 3]

std_res = pd.DataFrame(columns=["Scenario", "Demog", "Select", "iter", "RMSE"])

for i in range(3):
    axs[i + 1][0].plot(range(1, 101), Ne_mdls[i], color="black")

for j in range(3):
    axs[0][j + 1].plot(range(1, 101), s_mdls[j]["s"], color="black")

std_res = []

for i in range(3):
    for j in range(3):
        res = []
        for seed in range(N_SIMULATIONS):
            this_res = sim_and_fit(
                s_mdls[j],
                seed=12345 + seed,
                lam=10 ** optimal_lambda[j],
                Ne=Ne_mdls[i],
                em_iterations=N_EM_ITERATIONS,
            )
            res.append(this_res)
            rmse = compute_rmse(this_res["s_hat"], s_mdls[i]["s"])
            std_res.append(
                {
                    "s_mdl": s_names[i],
                    "Ne_mdl": Ne_names[i],
                    "iter": seed,
                    "sampling": "100-10",
                    "rmse": rmse,
                }
            )
        x, y = zip(*[(range(len(rr["s_hat"])), rr["s_hat"]) for rr in res])
        plot_summary(axs[i + 1][j + 1], x, y, s_mdls[j]["s"])

std_res = pd.DataFrame(std_res)

fig.savefig("plots/Simulate1-10x.pdf")
