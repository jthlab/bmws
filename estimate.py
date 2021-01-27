#!/usr/bin/env python
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.stats

from common import xform, Observation, PosteriorDecoding
from fusedlasso import fusedlasso
from hmm import forward_backward


import logging

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnums=(2, 3, 5, 6))
@jax.value_and_grad
def obj(x_s, h, times, Ne, obs, d, M, lam_):
    s = xform(x_s)
    return (
        -forward_backward(times, s, h, Ne, obs, d, M, True)
        + lam_ * (jnp.diff(s) ** 2).sum()
    )


def estimate(
    data: List[Observation],
    h: float = 0.5,
    lam_: float = 1.0,
    d: int = 100,
    M: int = 100,
    ell1: bool = False,
):
    # sort everything in ascending order of time.
    Ne, obs, times = _prep_data(data)
    T = len(times)
    x0 = np.zeros(T - 1)
    args = (
        np.full_like(x0, h),
        times,
        Ne,
        obs,
        d,  # d
        M,  # M
    )
    if ell1:
        optimizer = fusedlasso
        args += (0.0,)
        options = {"lam": lam_}
    else:
        optimizer = "BFGS"
        args += (lam_,)
    options = {}
    i = 0

    def shim(x, *args):
        nonlocal i
        logger.debug("x=%s", x)
        ret = obj(x, *args)
        logger.debug("i=%d x=%s ret=%s", i, x, ret)
        i += 1
        return ret

    res = scipy.optimize.minimize(
        shim, x0, jac=True, args=args, method=optimizer, options=options
    )
    logger.debug("Optimization result: %s", res)
    return {"t": times, "s": xform(res.x), "obs": obs}


def _prep_data(data):
    times, Ne, obs = zip(
        *sorted([(ob.t, ob.Ne, (ob.sample_size, ob.num_derived)) for ob in data])[::-1]
    )
    obs = np.array(obs)
    if len(times) != len(set(times)):
        raise ValueError("times should be distinct")
    return Ne, obs, times


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
    gamma, discr = forward_backward(
        times, np.array(s), np.array(h), Ne, obs, M, d, False
    )
    assert np.allclose(gamma.sum(1), 1.0, atol=1e-4)
    G = gamma / gamma.sum(
        1, keepdims=True
    )  # numerical error may cause columns to not quite sum to one.
    return PosteriorDecoding(t=times, gamma=G, discretizations=discr.untree())
