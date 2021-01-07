#!/usr/bin/env python

import itertools as it
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

from bws import T_bws
from common import f_sh, midpoints
from fusedlasso import fusedlasso


def _scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in zip(*xs):
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


_scan = jax.lax.scan


def loglik(
    s: np.ndarray, h: np.ndarray, p_obs: np.ndarray, grid: np.ndarray, Ne: float
) -> Tuple[np.ndarray, np.ndarray]:
    """likelihood of allele frequency trajectory with selection.

    Args:
        s: selection coefficient at generation t=1,...,T.
        h: dominance parameter at t=1,...,T.
        p_obs: probability of data (array of shape [T, D])
        grid: discretization grid for allele frequencies, length [D + 1].
        Ne: (diploid) effective population size.

    Returns:
        log likelihood of data.
    """

    # forward pass
    def fwd(last_alpha, tup_i):
        p_obs_t, s_t, h_t = tup_i
        trans_t = T_bws(grid, Ne, s_t, h_t)
        alpha = (last_alpha @ trans_t) * p_obs_t
        c = alpha.sum()
        alpha = alpha / c
        return alpha, c

    alpha0 = p_obs[0]
    c0 = alpha0.sum()
    alpha0 /= c0
    alpha, c = _scan(fwd, alpha0, (p_obs[1:], s, h))
    c = jnp.concatenate([c0[None], c])
    return jnp.log(c).sum()


def xform(x):
    # ensure that 1+s, 1+sh > 0
    return -1.0 + jnp.exp(x)


@jax.jit
def obj(x, s_mode, fixed_val, p_obs, grid, Ne, lam_):
    sh = xform(x)
    z = jnp.full_like(sh, fixed_val)
    s, h = [jax.lax.cond(u, lambda _: sh, lambda _: z, None) for u in (s_mode, ~s_mode)]
    return -loglik(s, h, p_obs, grid, Ne) + lam_ * (jnp.diff(sh) ** 2).sum()


grad = jax.jit(jax.grad(obj))


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


def fit_full(
    obs: np.ndarray, #observations at each generatin (including 0 observations)
    size: np.ndarray, #Number of observations at each generation (can be 0)
    lam_: float,
    s_mode: bool = True,
    fixed_val: float = 0.5,
    ell1: bool = False,
    D: int = 100,
    Ne: int = 1000
):
    T = len(obs)  # number of time points
    grid = np.linspace(0, 1, D + 1)

    # Calculate emission probabilities
    p_obs = np.ones((T, D + 2))  # two extra states to account for fixation
    f = np.concatenate([midpoints(grid), [0.0], [1.0]])
    p_obs[::] = scipy.stats.binom.pmf(obs[:, None], size[:, None], f[None, :])

    # Perform optimization
    def _obj(*args):
        ret = obj(*args)
        return tuple(map(np.array, ret))

    args = (s_mode, fixed_val, p_obs, grid, Ne)
    if ell1:
        optimizer = fusedlasso
        args += (0.0,)
        options = {"lam": lam_}
    else:
        optimizer = "BFGS"
        args += (lam_,)
        options = {}
    x0 = jnp.zeros(T - 1)
    res = scipy.optimize.minimize(
        obj, x0, jac=grad, args=args, method=optimizer, options=options
    )
    return {"x": xform(res.x), "obs": obs}

def sim_full(
    mdl: Dict,
    seed: int,
    D: int = 100,
    Ne: int = 1000,
    n: int = 100,  # sample size
    d: int = 10  # sampling interval
):
    T = len(mdl["s"]) + 1  # number of time points
    rng = np.random.default_rng(seed)
    af = sim_wf(Ne, mdl["s"], mdl["h"], mdl["f0"], rng)
    obs=np.zeros(T, dtype=int)
    size=np.zeros(T, dtype=int)
    obs[::d] = rng.binomial(n, af[::d])  # sample n haploids every d generations
    size[::d] = n
    return obs, size

def sim_and_fit(
    mdl: Dict,
    seed: int,
    lam_: float,
    s_mode: bool = True,
    fixed_val: float = 0.5,
    ell1: bool = False,
):
    # Parameters
    T = len(mdl["s"]) + 1  # number of time points
    D = 100  # density of discretization
    Ne = 1e4  # effective population size
    n = 100  # sample size
    d = 10  # sampling interval
    grid = np.linspace(0, 1, D + 1)

    # Simulate true trajectory
    rng = np.random.default_rng(seed)
    af = sim_wf(Ne, mdl["s"], mdl["h"], mdl["f0"], rng)
    obs = rng.binomial(n, af[::d])  # sample n haploids every d generations

    # Calculate emission probabilities
    p_obs = np.ones((T, D + 2))  # two extra states to account for fixation
    f = np.concatenate([midpoints(grid), [0.0], [1.0]])
    p_obs[::d] = scipy.stats.binom.pmf(obs[:, None], n, f[None, :])

    # Perform optimization
    def _obj(*args):
        ret = obj(*args)
        return tuple(map(np.array, ret))

    args = (s_mode, fixed_val, p_obs, grid, Ne)
    if ell1:
        optimizer = fusedlasso
        args += (0.0,)
        options = {"lam": lam_}
    else:
        optimizer = "BFGS"
        args += (lam_,)
        options = {}
    x0 = jnp.zeros(T - 1)
    res = scipy.optimize.minimize(
        obj, x0, jac=grad, args=args, method=optimizer, options=options
    )
    return {"x": xform(res.x), "obs": obs}
