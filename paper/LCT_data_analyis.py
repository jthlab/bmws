#The same as the LCT_data_analysis notebook, but in a script...

import estimate, sim
import matplotlib.pyplot as plt
import numpy as np
from betamix import sample_paths, BetaMixture
from plotting import plot_ci
from sim import sim_and_fit
from scipy.stats import beta
from math import exp, log

N_BOOSTRAP_REPS=1000

def load_data(file):
    oo = np.loadtxt(file)
    return [oo[:, 1], oo[:, 0]]

genes = ["LCT", "SLC45A2", "DHCR7", "HERC2"]

obs = {x: np.loadtxt("data/Britain_" + x + ".txt").astype(int)[::-1] for x in genes}

fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(12, 4), sharey="row")

log10_lam=4.5


for i,(gen,data) in enumerate(obs.items()):
    L = len(data)
    Ne = np.full(L - 1, 1e4)
    gr = exp(log(100) / (L))
    Ne=np.array([round(10000 * gr ** (10 * int(x / 10))) for x in range(L - 1)])
    s_hat, prior = estimate.estimate_em(data, Ne, lam=10 ** log10_lam, em_iterations=3)
    paths, _ = sample_paths(s_hat, Ne, data, N_BOOSTRAP_REPS, prior=prior)
    
    nk=[(i, d[0]) for i, d in enumerate(data) if d[0] > 0]
    n=[d[1] for d in nk]
    k=[d[0] for d in nk]
    
    res = []
    for seed in range(N_BOOSTRAP_REPS):
        res.append(sim_and_fit({"s":s_hat}, seed=seed, lam=10 ** log10_lam, n=n, k=k, Ne=Ne, af=paths[seed]))
    
    axs[0][i].plot(s_hat, label=gen)
    plot_ci(axs[0][i], [tuple(range(L-1))], np.array([r["s_hat"] for r in res]))

    axs[1][i].plot(np.mean(paths, axis=0), label=gen)
    plot_ci(axs[1][i], [tuple(range(L))], paths)
    
    axs[0][i].invert_xaxis()
    axs[1][i].invert_xaxis()
    axs[0][i].set_xlabel("Generations before present")
    axs[1][i].set_xlabel("Generations before present")
    
axs[0][0].set_ylabel("Selection coefficient")
axs[1][0].set_ylabel("Allele frequency")
[axs[0][i].set_title(g) for i, g in enumerate(genes)]

fig.savefig("plots/Figure5.pdf")
