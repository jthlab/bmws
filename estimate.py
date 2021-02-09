#!/usr/bin/env python
import logging
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.stats

from common import Observation, PosteriorDecoding
from fusedlasso import fusedlasso
from hmm import forward_backward, stochastic_traceback

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnums=(2, 5, 6))
@jax.value_and_grad
def obj(s, h, times, Ne, obs, M, lam_):
    return (
        -forward_backward(s, h, times, Ne, obs, M, True)["loglik"]
        + lam_ * (jnp.diff(s) ** 2).sum()
    )


def estimate(
    data: List[Observation],
    h: float = 0.5,
    lam_: float = 1.0,
    M: int = 100,
    ell1: bool = False,
):
    # sort everything in ascending order of time.
    Ne, obs, times = _prep_data(data)
    T = len(times)
    x0 = np.zeros(T - 1)
    args = (np.full_like(x0, h), times, Ne, obs, M)  # M
    kwargs = {}
    if ell1:
        optimizer = fusedlasso
        args += (0.0,)
        options = {"lam": lam_}
    else:
        optimizer = "L-BFGS-B"
        args += (lam_,)
        kwargs["bounds"] = [[-0.2, 0.2]] * len(x0)
    options = {}
    i = 0

    def shim(x, *args):
        nonlocal i
        logger.debug("x=%s", x)
        ret = obj(x, *args)
        logger.debug("i=%d x=%s ret=%s", i, x, ret)
        i += 1
        return tuple(np.array(x, dtype=np.float64) for x in ret)

    res = scipy.optimize.minimize(
        shim, x0, jac=True, args=args, method=optimizer, options=options, **kwargs
    )
    logger.debug("Optimization result: %s", res)
    return {"t": times, "s": res.x, "obs": obs}


def _prep_data(data):
    times, Ne, obs = zip(
        *sorted([(ob.t, ob.Ne, (ob.sample_size, ob.num_derived)) for ob in data])[::-1]
    )
    if len(times) != len(set(times)):
        raise ValueError("times should be distinct")
    obs = np.array(obs)
    return np.array(Ne), obs, times


def sample_paths(
    data: List[Observation],
    s: np.ndarray,
    h: float = 0.5,
    M: int = 64,
    num_paths: int = 1,
):
    Ne, obs, times = _prep_data(data)
    assert len(times) == len(s) + 1 == len(h) + 1
    res = forward_backward(np.array(s), np.array(h), times, Ne, obs, M, True)
    seed = jnp.arange(num_paths)
    paths = jax.vmap(stochastic_traceback, in_axes=(None, None, 0))(
        res["alpha"], res["T"], seed
    )
    return jnp.take_along_axis(res["hidden_states"].T, paths, axis=0) / res["Ne"][None]


def posterior_decoding(
    data: List[Observation], s: np.ndarray, h: float = 0.5, M: int = 64, d: int = 64
):
    """Find the posterior decoding given model parameters.

    Args:
        data: The observed data.
        s: The selection coefficients _between_ each time point.
        h: The dominance coefficients _between_ each time point.
        M: The number of bins used to discretize the allele frequency space.

    Returns:
        Posterior decoding matrix gamma.

    Notes:
        Assuming that T observations are made at time points t=[t[0],...,t[T-1]], the dimensions of
        gamma will be [M, T-1].

    """
    Ne, obs, times = _prep_data(data)
    assert len(times) == len(s) + 1 == len(h) + 1
    res = forward_backward(np.array(s), np.array(h), times, Ne, obs, M, False)
    return PosteriorDecoding(
        t=times,
        gamma=res["gamma"],
        Ne=res["Ne"],
        # B=res["B"],
        # T=res["log_T"],
        hidden_states=res["hidden_states"],
    )
