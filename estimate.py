#!/usr/bin/env python
import logging
from functools import partial
from typing import List, Union

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.stats
from jax import jit, vmap, value_and_grad
from jax.experimental.host_callback import id_print

import betamix
import hmm
from common import Observation, PosteriorDecoding
from fusedlasso import fusedlasso
from hmm import forward_backward, stochastic_traceback
from betamix import loglik

logger = logging.getLogger(__name__)


def _obj(s, Ne, obs, M, lam):
    ll = loglik(s, Ne, obs, M)
    return -ll + lam * (jnp.diff(s) ** 2).sum()


obj = jit(value_and_grad(_obj), static_argnums=(3,))


def estimate(
    obs,
    Ne,
    lam: float = 1.0,
    M: int = 100,
    solver_options: dict = {},
):
    args = (Ne, obs, M)
    x0 = np.zeros(len(obs) - 1)
    kwargs = {}
    optimizer = "L-BFGS-B"
    args += (lam,)
    kwargs["bounds"] = [[-0.2, 0.2]] * len(x0)
    i = 0

    def shim(x, *args):
        nonlocal i
        ret = tuple(np.array(y, dtype=np.float64) for y in obj(x, *args))
        logger.debug("i=%d f=%f |df|=%f", i, ret[0], np.linalg.norm(ret[1]))
        i += 1
        return ret

    # from jax.experimental import optimizers
    #
    # opt_init, opt_update, get_params = optimizers.adagrad(
    #     solver_options.get("learning_rate", 1.0)
    # )
    # opt_state = opt_init(x0)
    #
    # def step(step, opt_state):
    #     value, grads = obj(get_params(opt_state), *args)
    #     opt_state = opt_update(step, grads, opt_state)
    #     return value, opt_state, grads
    #
    # for i in range(1000):
    #     value, opt_state, grads = step(i, opt_state)
    #     logger.debug("f=%f |grad|=%f", value, np.linalg.norm(grads))
    #
    # logger.debug("value=%f grad=%s", value, grads)
    #
    # return get_params(opt_state)

    res = scipy.optimize.minimize(
        shim,
        x0,
        jac=True,
        args=args,
        method=optimizer,
        options=solver_options,
        **kwargs
    )
    logger.debug("Optimization result: %s", res)
    return res.x


def _prep_data(data):
    times, Ne, obs = zip(
        *sorted([(ob.t, ob.Ne, (ob.sample_size, ob.num_derived)) for ob in data])[::-1]
    )
    if len(times) != len(set(times)):
        raise ValueError("times should be distinct")
    obs = np.array(obs)
    assert np.all(obs[:, 1] <= obs[:, 0])
    assert np.all(obs[:, 1] >= 0)
    return np.array(Ne), obs, times


def sample_paths(
    s: np.ndarray,
    obs,
    Ne,
    k: int,
    M: int = 100,
):
    return betamix.sample_paths(s, obs, Ne, k, M)


# def posterior_decoding(
#     data: List[Observation], s: np.ndarray, discr_or_M: Union[np.ndarray, int]
# ):
#     """Find the posterior decoding given model parameters.
#
#     Args:
#         data: The observed data.
#         s: The selection coefficients _between_ each time point.
#         discr_or_M: A discretization (created by hmm.make_discr), or the number of bins used to discretize the allele
#             frequency space.
#
#     Returns:
#         Posterior decoding matrix gamma.
#
#     Notes:
#         Assuming that T observations are made at time points t=[t[0],...,t[T-1]], the dimensions of
#         gamma will be [M, T-1].
#
#     """
#     Ne, obs, times = _prep_data(data)
#     assert len(times) == len(s) + 1
#     if isinstance(discr_or_M, int):
#         M = discr_or_M
#         discr = hmm.make_discr(Ne, M)
#     else:
#         discr = discr_or_M
#     res = forward_backward(np.array(s), times, Ne, obs, discr, False)
#     return PosteriorDecoding(t=times, gamma=res["gamma"], hidden_states=discr, Ne=Ne)
