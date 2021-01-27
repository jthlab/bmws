"HMM-related functions"
from functools import partial
from typing import Tuple, Union

import jax
from jax import numpy as jnp
from jax.scipy.special import xlogy, gammaln
import numpy as np

from common import f_sh, poisson_cdf, binom_pmf, Discretization

vf_sh = jnp.vectorize(f_sh)


def trans(disc1, disc2, s: float, h: float) -> np.ndarray:
    """transition matrix for allele frequency under selection

    Params:
        disc1: starting discretization
        disc2: ending discretization
        s: Selection coefficient
        h: Dominance coefficient

    Returns:
        Transition matrix of dimension ...

    Notes:
        The discretization for allele frequency is selected as follows. The first and last `c` points
        are chosen discretely. The remaining `M` points are equally spaced between `c` and `Ne - c`.
        The resulting discretization looks like:

            {0, 1, 2, ..., c - 1, [c, c + (Ne - c) / M), [(Ne - c) / M, 2 (Ne - c) / M), ...,
                [(M - 1) * (Ne - c) / M, Ne - c), Ne - c + 1, ..., Ne - 1, Ne}

    """
    low1, mid1, high1, Ne1 = disc1
    low2, mid2, high2, Ne2 = disc2

    # low1, mid1, high1 = id_print((low1, mid1, high1), what="disc1")
    # low2, mid2, high2 = id_print((low2, mid2, high2), what="disc2")

    # Low
    # The extreme states {0,1,...,c-1} U {Ne-c+1,...,Ne} are assumed Poisson distributed.
    # we skip the first state x=0 because it causes problems with nans.
    mu_low = vf_sh(low1[1:, None] / Ne1, s, h)
    T_11 = jax.scipy.stats.poisson.pmf(low2[None, :], mu_low)
    # low -> medium
    T_12 = jnp.diff(poisson_cdf(mid2[None, :], mu_low), axis=1)
    # low -> high  (improbable unless Ne is tiny)
    T_13 = jax.scipy.stats.poisson.pmf(high2[None, :], mu_low)

    # Medium
    mu_mid = vf_sh((mid1[:-1, None] + mid1[1:, None]) / 2 / Ne1, s, h)
    sd_mid = jnp.sqrt(mu_mid * (1 - mu_mid) / Ne1)
    # Medium -> low
    T_21 = binom_pmf(low2[None, :], Ne2, mu_mid)
    # Medium -> medium
    T_22 = jnp.diff(
        jax.scipy.stats.norm.cdf(mid2[None, :] / Ne2, loc=mu_mid, scale=sd_mid)
    )
    T_23 = binom_pmf(high2[None, :], Ne2, mu_mid)

    # High (mirror image of low) -- skip the last state for same reason
    mu_high = vf_sh(high1[:-1, None] / Ne1, s, h)
    # high -> low
    T_31 = jax.scipy.stats.poisson.pmf(low2[None, :], mu_high)
    # high -> medium
    T_32 = jnp.diff(poisson_cdf(mid2[None, :], mu_high), axis=1)
    # high -> high  (improbable unless Ne is tiny)
    T_33 = jax.scipy.stats.poisson.pmf(high2[None, :], mu_high)

    T0 = jnp.block(
        [
            [T_11, T_12, T_13],
            [T_21, T_22, T_23],
            [T_31, T_32, T_33],
        ]
    )
    # add in the first and last states that we skipped
    N = T0.shape[0] + 2
    ret = jnp.concatenate([jnp.eye(1, N, 0), T0, jnp.eye(1, N, N - 1)]).astype(
        jnp.float64
    )
    ret = jnp.maximum(1e-8, ret)
    ret /= ret.sum(1, keepdims=True)
    # ret = id_print(ret, what="trans")
    return ret


def test_trans():
    T = trans(20, 500, 1e3, 0.01, 0.5)
    assert np.isfinite(T).all()
    np.testing.assert_allclose(T.sum(1), 1.0, atol=1e-4)


def _scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    keys = list(xs.keys())
    n = len(xs[keys[0]])
    for i in range(n):
        d = {}
        for k in keys:
            if isinstance(xs[k], Discretization):
                d[k] = Discretization(xs[k].low[i], xs[k].mid[i], xs[k].high[i])
            else:
                d[k] = xs[k][i]
        carry, y = f(carry, d)
        ys.append(y)
    return carry, jnp.stack(ys)


@partial(jax.jit, static_argnums=1)
def _jmp(A, k):
    return jnp.linalg.matrix_power(A, k)


@partial(jax.jit, static_argnums=2)
def _fb(T, B, loglik):
    alpha0 = B[0]
    c0 = alpha0.sum()
    alpha0 /= c0

    def _fwd(last_alpha, params):
        alpha = (last_alpha @ params["T"]) * params["B"]
        c = alpha.sum()
        alpha = alpha / c
        if loglik:
            return alpha, c
        return alpha, (alpha, c)

    params = {"T": T, "B": B[1:]}
    res = jax.lax.scan(_fwd, alpha0, params)

    if loglik:
        _, c = res
        return jnp.log(c0) + jnp.log(c).sum()

    _, (alpha, c) = res
    alpha = jnp.concatenate([alpha0[None], alpha])

    beta0 = jnp.ones_like(alpha0)
    # backward pass
    def _bwd(last_beta, params):
        beta = params["T"] @ (last_beta * params["B"]) / params["c"]
        return (beta,) * 2

    params["c"] = c
    _, beta = jax.lax.scan(_bwd, beta0, params, reverse=True)
    beta = jnp.concatenate([beta, beta0[None]])
    gamma = alpha * beta
    return gamma


def forward_backward(
    times: Tuple[int, ...],
    s: np.ndarray,
    h: np.ndarray,
    Ne: Tuple[int, ...],
    obs: np.ndarray,
    M: int = 100,
    d: int = 20,
    only_loglik: bool = False,
) -> Union[float, Tuple[np.ndarray, Discretization]]:
    """likelihood of allele frequency trajectory with selection.

    Args:
        times: [T] number of generations before present when each observation was sampled. Assumed sorted in *descending*,
            order, i.e. the oldest observations come first. See Notes.
        s: [T - 1] selection coefficient at each time point
        h: [T - 1] dominance parameter at each time point
        Ne: [T - 1] diploid effective population size at each time point
        obs [T, 2]: (sample size, # derived alleles) observed at each time point. missing data can
            be encoded by setting sample_size == 0.
        M: discretization parameter -- state space will be discretized into this many bins.
        d: Endpoint discretization -- the first and last d bins will correspond to the discrete bins
            {0, ..., d - 1} and {Ne - d + 1, ..., Ne}. (Setting d=0 disables this.)
        only_loglik: If true, only compute the log likelihood. Otherwise, compute the full posterior decoding.

    Returns:
        If only_loglik = True, return the log likelihood. Otherwise, return the posterior decoding matrix.

    Notes:
        With X denoting the unobserved allele frequency, and dt[i] = times[i + 1] - times[i],
        the data model for the HMM is:


       (Ne[0])   h[0],s[0]                                   h[T - 2], s[T - 2]   (Ne[T - 1])
        X[0]   ----------->   X[1]  --- ... ---> X[T - 2]  ----------------------> X[T - 1]
         |       dt[0]          |                    |           dt[t - 2]             |
       obs[0]                obs[1]            obs[T - 2]                          obs[T - 1]
     (times[0])            (times[1]         (times[T - 2])                       (times[T - 1])

                         [Past ------------> present]

    """
    T = len(times)
    assert T == len(Ne) == len(obs)
    assert T - 1 == len(s) == len(h)
    Ne = jnp.array(Ne)

    dt = -np.diff(times)
    discr = jax.vmap(Discretization.factory, in_axes=(None, None, 0))(M, d, Ne)
    B = jax.vmap(lambda disc, ob: binom_pmf(ob[1], ob[0], disc.p), in_axes=(0, 0))(
        discr, obs
    )
    T0 = jax.vmap(trans)(discr.head, discr.head, s, h)
    T1 = jax.vmap(trans)(discr.head, discr.tail, s, h)
    T = jnp.array([_jmp(T0i, dti - 1) @ T1i for T0i, T1i, dti in zip(T0, T1, dt)])
    res = _fb(T, B, only_loglik)
    if only_loglik:
        return res
    return np.asarray(res), discr
