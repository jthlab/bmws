from typing import Dict, Union

import numpy as np

from common import Observation, f_sh
from estimate import estimate


def sim_wf(
    N: int, s: np.ndarray, h: np.ndarray, f0: int, rng: Union[int, np.random.Generator]
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
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    f = np.zeros(T + 1)
    f[0] = f0
    for t in range(1, T + 1):
        p = f_sh(f[t - 1], s[t - 1], h[t - 1])
        f[t] = rng.binomial(2 * N, p) / (2 * N)
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
    ell1: bool = False,
    Ne=1e4,  # effective population size
    n=100,  # sample size
    k=10,  # sampling interval
    **kwargs
):
    # Parameters
    T = len(mdl["s"]) + 1  # number of time points

    # Simulate true trajectory
    rng = np.random.default_rng(seed)
    af = sim_wf(Ne, mdl["s"], mdl["h"], mdl["f0"], rng)
    obs = rng.binomial(n, af[::k])  # sample n haploids every d generations

    data = [
        Observation(t=t, sample_size=n, num_derived=oo, Ne=Ne)
        for t, oo in zip(range(0, len(af), k), obs)
    ]

    return estimate(data, lam=lam, ell1=ell1, **kwargs)
