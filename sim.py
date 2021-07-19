from typing import Dict, Union

import numpy as np

from common import Observation, f_sh
from estimate import estimate, _prep_data
from betamix import BetaMixture
from scipy.stats import beta

def sim_wf(
    N: np.ndarray,
    s: np.ndarray,
    h: np.ndarray,
    f0: int,
    rng: Union[int, np.random.Generator],
):
    """Simulate T generations under wright-fisher model with population size 2N, where
    allele has initial frequency f0 and the per-generation selection coefficient is
    s[t], t=1,...,T.

    Returns:
        Vector f = [f[0], f[1], ..., f[T]] of allele frequencies at each generation.
    """
    assert 0 <= f0 <= 1
    T = len(s)
    assert T == len(h)
    assert T == len(N)

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    f = np.zeros(T + 1)
    f[0] = f0
    for t in range(1, T + 1):
        p = f_sh(f[t - 1], s[t - 1], h[t - 1])
        f[t] = rng.binomial(2 * N[t - 1], p) / (2 * N[t - 1])
    return f


def sim_full(
    mdl: Dict,
    seed: int,
    D: int = 100,
    Ne: int = 1000,
    n: int = 100,  # sample size
    d: int = 10,  # sampling interval
):
    T = len(mdl["s"]) + 1  # number of time points
    rng = np.random.default_rng(seed)
    af = sim_wf(Ne, mdl["s"], mdl["h"], mdl["f0"], rng)
    obs = np.zeros(T, dtype=int)
    size = np.zeros(T, dtype=int)
    obs[::d] = rng.binomial(n, af[::d])  # sample n haploids every d generations
    size[::d] = n
    return obs, size


def sim_and_fit(
    mdl: Dict,
    seed: int,
    lam: float,
    Ne=1e4,  # effective population size
    n=100,  # sample size - either an integer, or an iterable of sizes
    k=10,  # sampling interval - either an integer, or an iterable of sampling times.
    Ne_fit=None, # Ne to use for estimation (if different to that used for simulation)
    **kwargs
):
    # Parameters
    T = len(mdl["s"]) + 1  # number of time points

    # Set up population size.

    if isinstance(Ne, int) or isinstance(Ne, float):
        Ne = np.array([Ne] * (T - 1))
    else:
        Ne = np.array(Ne)

    if not Ne_fit:
        Ne_fit=Ne
    else: 
        Ne_fit=np.array(Ne_fit)
        
    # Set up sampling scheme
    sample_times, sample_sizes = k, n
    if isinstance(n, int) and isinstance(k, int):
        sample_times = range(T)[::k]
        sample_sizes = [n for x in sample_times]
        
    # Simulate true trajectory
    rng = np.random.default_rng(seed)
    # simulate the wright-fisher model. all parametetr vectros are reversed so that times runs from past to present.
    af = sim_wf(Ne[::-1], mdl["s"][::-1], mdl["h"][::-1], mdl["f0"], rng)[::-1]
    obs = np.zeros([T, 2])
    obs[sample_times, :] = [
        (n, rng.binomial(n, af[t])) for n, t in zip(sample_sizes, sample_times)
    ]

    #setup prior
    M = 200
    prior = BetaMixture.uniform(M)
    f = np.mean([x[1] / x[0] for x in obs if x[0] > 0])
    b = 4
    a = b * f / (1 - f)
    dens = [beta.pdf((x + 1) / (M + 1), a, b) for x in range(M)]
    total = np.sum(dens)
    dens = [x / total for x in dens]
    prior = prior._replace(log_c=np.log(dens))
    
    s_hat = estimate(obs, Ne_fit, lam=lam, prior=prior, **kwargs)
    return {"s_hat": s_hat, "obs": obs, "Ne": Ne, "true_af": af}
