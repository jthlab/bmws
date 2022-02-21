"HMM-related functions"
from typing import Dict, Tuple

import jax
from jax.experimental.host_callback import id_print

jax.config.update("jax_enable_x64", True)
import numpy as np
from jax import numpy as jnp
from jax.scipy.special import xlogy

from common import binom_pmf, f_sh, matpow_ub

vf_sh = jnp.vectorize(f_sh)


def trans_exact(s, discr1, Ne1, discr2, Ne2):
    # discr1 and k2 represent the discretizations before and after mating. depending on the respective sizes,
    # say Ne1, Ne2, they can either be discrete or continuous (intervals).
    # We can always assume that Ne2 < M, implying that k2 is discrete -- else we wouldn't be calling this function.
    # discr1 could be continuous. If so, it's endpoint will be 1. Otherwise, its endpoint will be Ne1 > 1.
    x0 = jnp.where(discr1 > Ne1, 0.0, discr1) / Ne1
    p = f_sh(x0, s, 0.5)
    k = discr2
    # there can be >1 zero state due to the way we discretize. this un
    T = binom_pmf(k[None, :], Ne2, p[:, None])
    T = jnp.where(discr1[:, None] > Ne1, 0.0, T)
    return T


def trans_infinite(s, dt, discr1, discr2):
    # ensure that trans_infinite is not called for dt=0. in that case,
    # in should just return the identity matrix.
    return jax.lax.cond(
        dt == 0,
        lambda _: jnp.eye(len(discr1)),
        lambda _: _trans_infinite(s, dt, discr1, discr2),
        None,
    )


def _trans_infinite(s, dt, discr1, discr2):
    x0 = discr1[1:-1] / discr1[-1]
    Ne2 = discr2[-1]

    mu = x0 / (x0 + (1 - x0) * jnp.exp(-s * dt))

    def variance(s):
        return (
            mu ** 2
            * (2 + s)
            * x0
            * (1 - x0)
            / s
            * (
                2 * x0 * (1 - x0) * s * dt
                + x0 ** 2 * jnp.expm1(s * dt)
                - (1 - x0) ** 2 * jnp.expm1(-s * dt)
            )
        )

    def variance_asymp(s):
        return -2 * dt * (-1 + x0) * x0 + dt * s * (-1 + x0) * x0 * (
            -1 + dt * (-3 + 6 * x0)
        )

    sigma2 = jax.lax.cond((-1e-8 < s) & (s < 1e-8), variance_asymp, variance, s)

    # sigma2, _ = id_print((sigma2, (dt, s, mu)), what="sigma2")

    x1 = 0.5 * (discr2[1:] + discr2[:-1]) / Ne2
    P = jax.scipy.stats.norm.cdf(
        x1[None], loc=mu[:, None], scale=jnp.sqrt(sigma2[:, None] / Ne2)
    )
    dP = jnp.concatenate([P[:, :1], jnp.diff(P, axis=1), 1.0 - P[:, -1:]], axis=1)
    M = dP.shape[1]
    T = jnp.concatenate([jnp.eye(1, M, 0), dP, jnp.eye(1, M, M - 1)])
    return T


def fb(T, B, loglik):
    alpha0 = B[0][None]
    c0 = alpha0.sum()
    alpha0 /= c0

    def _fwd(last_alpha, params):
        assert last_alpha.ndim == 2
        assert params["T"].ndim == 2
        assert params["B"].ndim == 1
        assert last_alpha.shape[0] == 1
        assert (
            params["T"].shape[0]
            == params["T"].shape[1]
            == last_alpha.shape[1]
            == params["B"].shape[0]
        )
        # params = id_print(params, what="params")
        alpha = (last_alpha @ params["T"]) * params["B"][None, :]
        assert alpha.shape == last_alpha.shape
        c = alpha.sum()
        alpha /= c
        # alpha = id_print(alpha, what="alpha")
        # c = id_print(c, what="c")
        return alpha, (alpha[0], c)

    params = {"T": T, "B": B[1:]}
    res = jax.lax.scan(_fwd, alpha0, params)

    _, (alpha, c) = res
    alpha = jnp.concatenate([alpha0, alpha])
    c = jnp.concatenate([c0[None], c])
    # log_alpha, log_c = id_print((log_alpha, log_c), what="fwd")
    if loglik:
        return {"alpha": alpha, "c": c, "loglik": jnp.log(c).sum()}

    beta0 = jnp.ones_like(alpha0).T
    # backward pass
    def _bwd(last_beta, params):
        assert last_beta.ndim == 2
        assert params["T"].ndim == 2
        assert params["B"].ndim == 1
        assert params["c"].ndim == 0
        assert last_beta.shape[1] == 1
        assert (
            params["T"].shape[0]
            == params["T"].shape[1]
            == last_beta.shape[0]
            == params["B"].shape[0]
        )
        beta = params["T"] @ (last_beta * params["B"][:, None]) / params["c"]
        # beta = id_print(beta, what="alpha")
        return (beta, beta[:, 0])

    params["c"] = c[1:]
    _, beta = jax.lax.scan(_bwd, beta0, params, reverse=True)
    beta = jnp.concatenate([beta, beta0.T])
    gamma = alpha * beta
    return {"gamma": gamma}


def forward_backward(
    s: np.ndarray,
    times: Tuple[int, ...],
    Ne: Tuple[int, ...],
    obs: np.ndarray,
    M: int = 100,
    forward_only: bool = False,
) -> Dict[str, jnp.ndarray]:
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
    assert T - 1 == len(s)
    ub = int(1 + np.log2(max(times)))
    dt = -np.diff(times)  # number of generations between observations
    assert all(dt > 0)

    # Our discretization is fixed to M hidden states but can be different depending on whether
    # Ne is small or large. If Ne <= M then we simply run the exact W-F model on floor(Ne) individuals.
    # If Ne exceeds M then the transition function is approximated by the infinite population limit.
    # The reason for keeping the discretization size constant is that then we can use jax.vmap. This is more
    # efficient than using Python loops, which get statically unrolled during jit.
    # (However it can be inefficient if Ne << M.)

    def discr_range(Ne):
        # odd = 1 - M % 2
        # half = jnp.concatenate([jnp.zeros(1), jnp.geomspace(1, Ne / 2, M // 2)])[
        #     : -odd or None
        # ]
        # R_log = jnp.concatenate([half, Ne - half[-(2 - odd) :: -1]])
        # R_log, _ = id_print((R_log, Ne), what="discr_range")
        R_log = jnp.concatenate(
            [jnp.array([0.0]), jnp.linspace(1, Ne - 1, M - 1), jnp.array([Ne])]
        )
        assert len(R_log) == M + 1
        return R_log

    R_disc = 1.0 * np.arange(M + 1)

    def discr_exact(Ne):
        return R_disc

    def make_trans(d):
        # d = id_print(d, what="d")
        discr1, T0 = jax.lax.cond(
            d["Ne1"] <= M,
            lambda e: (
                e["exact"],
                matpow_ub(
                    trans_exact(d["s"], e["exact"], d["Ne1"], e["exact"], d["Ne1"]),
                    d["dt"] - 1,
                    ub,
                ),
            ),
            lambda e: (
                e["range"],
                trans_infinite(d["s"], d["dt"] - 1, e["range"], e["range"]),
            ),
            {"exact": R_disc, "range": discr_range(d["Ne2"])},
        )
        T1 = jax.lax.cond(
            d["Ne2"] <= M,
            lambda discr1: trans_exact(d["s"], discr1, d["Ne1"], R_disc, d["Ne2"]),
            lambda discr1: trans_infinite(d["s"], 1, discr1, discr_range(d["Ne2"])),
            discr1,
        )
        return jnp.maximum(1e-100, T0 @ T1), discr1

    T, discr = jax.lax.map(
        make_trans, {"s": s, "Ne1": Ne[:-1], "Ne2": Ne[1:], "dt": dt}
    )

    def make_obs(d):
        discr = jax.lax.cond(
            d["Ne"] <= M,
            lambda e: e["exact"],
            lambda e: e["range"],
            {"exact": discr_exact(d["Ne"]), "range": discr_range(d["Ne"])},
        )
        ss = discr <= d["Ne"]
        p = jnp.where(ss, discr / d["Ne"], 0.5)
        ret = binom_pmf(d["obs"][1], d["obs"][0], p)
        # ret = id_print(ret, what="ret")
        return jnp.maximum(jnp.where(ss, ret, 0.0), 1e-100), discr

    B, discr = jax.lax.map(make_obs, {"Ne": Ne, "obs": obs})

    # T = id_print(T, what="T")
    # B = id_print(B, what="B")

    ret = fb(T, B, forward_only)
    ret.update({"B": B, "T": T, "hidden_states": discr, "Ne": Ne})
    return ret


def stochastic_traceback(alpha, T, seed):
    log_alpha = jnp.log(alpha)
    log_T = jnp.log(T)

    def _f(carry, d):
        key, subkey = jax.random.split(carry["key"])
        log_p = d["log_T"][:, carry["x_last"]] + d["log_alpha"]
        x = jax.random.categorical(subkey, log_p)
        return {"key": key, "x_last": x}, x

    key, subkey = jax.random.split(jax.random.PRNGKey(seed))
    x0 = jax.random.categorical(subkey, log_alpha[-1])
    _, x = jax.lax.scan(
        _f,
        {"key": subkey, "x_last": x0},
        {"log_alpha": log_alpha[:-1], "log_T": log_T},
        reverse=True,
    )
    return jnp.concatenate([x, x0[None]])
