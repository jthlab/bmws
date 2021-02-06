"HMM-related functions"
import dataclasses
from functools import partial
from typing import Tuple, Union

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental.host_callback import id_print
from jax.scipy.special import xlogy

# id_print = lambda x, *args, **kwargs: x

from common import (
    f_sh,
    poisson_cdf,
    Discretization,
    binom_logpmf,
    binom_logcdf_cp,
    logexpm1,
    log_matmul,
    log_matpow_ub,
    safe_logdiff,
)

vf_sh = jnp.vectorize(f_sh)


def log_trans_exact(s, h, discr1, discr2):
    # discr1 and k2 represent the discretizations before and after mating. depending on the respective sizes,
    # say Ne1, Ne2, they can either be discrete or continuous (intervals).
    # We can always assume that Ne2 < M, implying that k2 is discrete -- else we wouldn't be calling this function.
    # discr1 could be continuous. If so, it's endpoint will be 1. Otherwise, its endpoint will be Ne1 > 1.
    p = f_sh(discr1 / discr1[-1], s, h)
    k = discr2
    Ne2 = discr2[-1]
    ret = binom_logpmf(k[None, :], Ne2, p[:, None])
    # ret = id_print(ret, what="log_trans_exact()")
    return ret


def log_trans_discr(s: float, h: float, discr1, discr2) -> np.ndarray:
    # as above, discr1 and discr2 represent the discretizations before and after mating. depending on the respective sizes,
    # say Ne1, Ne2, they can either be discrete or continuous (intervals).
    # We can always assume that Ne2 > M, implying that k2 is continuous -- else we wouldn't be calling this function.
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
    #
    # when the allele is at low frequency we model it as Poisson draws. Low frequency is defined somewhat arbitrarily
    # to be bottom 5% of the discretization. This could probably be improved. It's better to set it too high than too low,
    # since CLT behavior saves us in that case.
    # s, h, discr1, discr2 = id_print((s, h, discr1, discr2), what="log_trans_discr()")
    Ne1 = discr1[-1]
    # np < 5, n > 100 => p <= .2
    L = int(0.05 * len(discr1))
    assert L > 0
    p0 = discr1 / discr1[-1]
    p1 = vf_sh(p0, s, h)
    # p0, p1 = id_print((p0, p1), what="p0p1")
    lam_low = p1[:L] * Ne1
    c_pois = poisson_cdf(discr2[None, :], lam_low[:, None])
    # c_pois = Ne1 * jnp.ones_like(c_pois)
    T_1 = jnp.concatenate(
        [jnp.log(c_pois[:, :1]), safe_logdiff(c_pois, axis=1)], axis=1
    )

    mu_mid = p1[L : len(p1) - L]
    # mu_mid = id_print(mu_mid, what="mu_mid")
    Ne2 = discr2[-1]
    _T_22 = jnp.concatenate(
        [
            jnp.full_like(mu_mid[:, None], -jnp.inf),
            binom_logcdf_cp(discr2[None, :], Ne2, mu_mid[:, None]),
        ],
        axis=1,
    )
    # _T_22 = id_print(_T_22, what="T_22")
    log_p0 = _T_22[:, :-1]
    log_p1 = _T_22[:, 1:]
    # log(p1 - p0) = log(exp(log_p1) - exp(log_p0)) = log_p1 + log(1-exp(log_p0 - log_p1))
    # in some cases the camp-paulson approximation breaks down and we get log_p0>log_p1
    # only seems to happen when both p0, p1 ~= 0. so we just truncate.
    z = log_p1 <= log_p0
    # also have to be careful about subtracting -inf from anything. this messes up all the grads.
    log_p1_minus_p0 = jnp.where(
        z, 1.0, log_p1 - jnp.where(jnp.isneginf(log_p0), log_p1, log_p0)
    )
    # log_p1_minus_p0, _ = id_print((log_p1_minus_p0, (log_p1, log_p0, z)), what="lp1mp0")
    T_2 = jnp.where(
        z,
        -jnp.inf,
        log_p1 + jnp.where(jnp.isneginf(log_p0), 0.0, logexpm1(log_p1_minus_p0)),
    )

    # High is the mirror image of low - the ancestral allele is approximately Poisson.
    lam_high = Ne1 * (1.0 - p1[len(p1) - L :])
    c_pois = poisson_cdf(Ne2 - discr2[None, ::-1], lam_high[:, None])
    # c_pois = id_print(c_pois, what="c_pois")
    T_3 = jnp.concatenate([jnp.log(c_pois[:, :1]), safe_logdiff(c_pois, 1)], axis=1)[
        :, ::-1
    ]
    T = [T_1, T_2, T_3]
    # T = id_print(T, what="T123")
    return jnp.concatenate(T)


# @partial(jax.jit, static_argnums=2)
def _fb(log_T, log_B, loglik):
    lse = jax.scipy.special.logsumexp
    log_alpha0 = log_B[0][None]
    log_c0 = lse(log_alpha0)
    log_alpha0 -= log_c0

    def _fwd(last_log_alpha, params):
        assert last_log_alpha.ndim == 2
        assert params["log_T"].ndim == 2
        assert params["log_B"].ndim == 1
        assert last_log_alpha.shape[0] == 1
        assert (
            params["log_T"].shape[0]
            == params["log_T"].shape[1]
            == last_log_alpha.shape[1]
            == params["log_B"].shape[0]
        )
        # params = id_print(params, what="params")
        log_alpha = (
            log_matmul(last_log_alpha, params["log_T"]) + params["log_B"][None, :]
        )
        assert log_alpha.shape == last_log_alpha.shape
        log_c = lse(log_alpha)
        log_alpha -= log_c
        # log_alpha, log_c = id_print((log_alpha, log_c), what="log_alpha")
        return log_alpha, (log_alpha[0], log_c)

    params = {"log_T": log_T, "log_B": log_B[1:]}
    res = jax.lax.scan(_fwd, log_alpha0, params)

    _, (log_alpha, log_c) = res
    log_alpha = jnp.concatenate([log_alpha0, log_alpha])
    log_c = jnp.concatenate([log_c0[None], log_c])
    # log_alpha, log_c = id_print((log_alpha, log_c), what="fwd")
    if loglik:
        return {"log_alpha": log_alpha, "log_c": log_c}

    log_beta0 = jnp.zeros_like(log_alpha0).T
    # backward pass
    def _bwd(last_log_beta, params):
        assert last_log_beta.ndim == 2
        assert params["log_T"].ndim == 2
        assert params["log_B"].ndim == 1
        assert params["log_c"].ndim == 0
        assert last_log_beta.shape[1] == 1
        assert (
            params["log_T"].shape[0]
            == params["log_T"].shape[1]
            == last_log_beta.shape[0]
            == params["log_B"].shape[0]
        )
        # params = id_print(params, what="params")
        log_beta = (
            log_matmul(params["log_T"], last_log_beta + params["log_B"][:, None])
        ) - params["log_c"]
        # log_beta = id_print(log_beta, what="log_beta")
        return (log_beta, log_beta[:, 0])

    params["log_c"] = log_c[1:]
    _, log_beta = jax.lax.scan(_bwd, log_beta0, params, reverse=True)
    log_beta = jnp.concatenate([log_beta, log_beta0.T])
    log_gamma = log_alpha + log_beta
    return {"log_gamma": log_gamma}


def forward_backward(
    s: np.ndarray,
    h: np.ndarray,
    times: Tuple[int, ...],
    Ne: Tuple[int, ...],
    obs: np.ndarray,
    M: int = 100,
    forward_only: bool = False,
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
        forward_only: If true, only compute the log likelihood. Otherwise, compute the full posterior decoding.

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
    ub = 1 + int(np.log2(max(times)))
    dt = -np.diff(times)  # number of generations between observations

    # Our discretization is fixed to M hidden states but can be different depending on whether
    # Ne is small or large. If Ne <= M then we simply simulate the exact W-F model on floor(Ne) individuals.
    # If Ne exceeds M then we use some approximations (see log_trans()).
    # The reason for keeping the discretization size constant is that then we can use jax.vmap. This is more
    # efficient than using Python loops, which get statically unrolled during jit.
    # (However it can be inefficient if Ne << M.)

    # construct log-spaced intervals from 1, ..., Ne/2 and then reflect. Special cases needed depending
    # on whether M is even or odd.
    R_disc = 1.0 * np.arange(M)

    def discr_range(Ne):
        odd = 1 - M % 2
        half = jnp.concatenate([jnp.zeros(1), jnp.geomspace(1, Ne / 2, M // 2)])[
            : -odd or None
        ]
        R_log = jnp.concatenate([half, Ne - half[-(2 - odd) :: -1]])
        assert len(R_log) == M
        # R_log, _ = id_print((R_log, Ne), what="discr_range")
        return R_log

    def discr_exact(Ne):
        ret = jnp.maximum(0, R_disc - (M - Ne))
        # ret = id_print(ret, what="discr_exact")
        return ret

    def make_trans(d):
        # d = id_print(d, what="d")
        d1, log_T0 = jax.lax.cond(
            d["Ne1"] <= M,
            lambda e: (
                e["exact"],
                log_trans_exact(d["s"], d["h"], e["exact"], e["exact"]),
            ),
            lambda e: (
                e["range"],
                log_trans_discr(d["s"], d["h"], e["range"], e["range"]),
            ),
            {"exact": discr_exact(d["Ne1"]), "range": discr_range(d["Ne2"])},
        )
        # _, log_T0 = id_print((d1, log_T0), what="(d1,log_T0")
        log_T1 = jax.lax.cond(
            d["Ne2"] <= M,
            lambda d1: log_trans_exact(d["s"], d["h"], d1, discr_exact(d["Ne2"])),
            lambda d1: log_trans_discr(d["s"], d["h"], d1, discr_range(d["Ne2"])),
            d1,
        )
        log_T0_t = log_matpow_ub(log_T0, d["t"] - 1, ub)
        trans = log_matmul(log_T0_t, log_T1)
        # trans, _ = id_print(
        #     (trans, jax.scipy.special.logsumexp(trans, 1)), what="lse(trans)"
        # )
        return trans, d1

    log_T, discr = jax.lax.map(
        make_trans, {"s": s, "h": h, "Ne1": Ne[:-1], "Ne2": Ne[1:], "t": dt}
    )

    def make_obs(d):
        discr = jax.lax.cond(
            d["Ne"] <= M,
            lambda e: e["exact"],
            lambda e: e["range"],
            {"exact": discr_exact(d["Ne"]), "range": discr_range(d["Ne"])},
        )
        return binom_logpmf(d["obs"][1], d["obs"][0], discr / d["Ne"])

    log_B = jax.lax.map(make_obs, {"Ne": Ne, "obs": obs})

    # log_T = id_print(log_T, what="log_T")
    # log_B = id_print(log_B, what="log_B")
    ret = _fb(log_T, log_B, forward_only)
    ret.update(
        {
            "log_T": log_T,
            "hidden_states": discr,
        }
    )
    return ret


def stochastic_traceback(log_alpha, log_T, seed):
    def _f(carry, d):
        key, subkey = jax.random.split(carry["key"])
        log_p = d["log_T"][:, carry["x_last"]] + d["log_alpha"]
        x = jax.random.categorical(subkey, log_p)
        return {"key": key, "x_last": x}

    key, subkey = jax.random.split(jax.random.PRNGKey(seed))
    x0 = jax.random.categorical(subkey, log_alpha[-1])
    _, x = jax.lax.scan(
        _f,
        {"key": subkey, "x_last": x0},
        {"log_alpha": log_alpha[:-1], "log_T": log_T},
        reverse=True,
    )
    return jnp.concatenate([x, x0[None]])
