import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from common import log_matpow
from hmm import forward_backward, trans_diffusion, trans_infinite

# testing

logging.getLogger("absl").setLevel(logging.DEBUG)


def test_fb():
    times = tuple(range(0, 200, 10))[::-1]
    T = len(times)
    s = np.zeros(len(times[:-1]))
    h = np.zeros_like(s)
    Ne = np.array([100.0] * T)
    n = np.random.randint(2, 20, size=T)
    derived = np.random.randint(0, n, size=T)
    obs = np.transpose([n, derived])
    obs[-1][:] = 10
    M = 21
    log_gamma = forward_backward(s, h, times, Ne, obs, M, False)
    np.testing.assert_allclose(logsumexp(log_gamma, 1), 0.0, atol=1e-4)
    np.testing.assert_allclose(logsumexp(log_gamma), np.log(T), atol=1e-4)


def test_fb_variable_Ne():
    times = tuple(range(0, 200, 10))[::-1]
    T = len(times)
    s = np.zeros(len(times[:-1]))
    h = np.zeros_like(s)
    Ne = np.array([100.0, 1e5, 1e6] * T)[:T]
    n = np.random.randint(2, 20, size=T)
    derived = np.random.randint(0, n, size=T)
    obs = np.transpose([n, derived])
    obs[-1][:] = 10
    M = 21
    log_gamma = forward_backward(s, h, times, Ne, obs, M, False)["log_gamma"]
    np.testing.assert_allclose(logsumexp(log_gamma, 1), 0.0, atol=1e-4)
    np.testing.assert_allclose(logsumexp(log_gamma), np.log(T), atol=1e-4)


def test_grad():
    Ne = np.array([1000.0, 50.0, 50.0, 1000.0, 1000.0])  # FIXME edge case where Ne == M
    # Ne[:] = 10.0
    times = tuple(np.arange(len(Ne)) * 10)[::-1]
    T = len(times)
    s = np.zeros(T - 1) + 0.01
    h = np.zeros_like(s) + 0.01
    n = np.random.randint(2, 20, size=T)
    derived = np.random.randint(0, n, size=T)
    obs = np.transpose([n, derived])
    M = 100

    args = (h, times, Ne, obs, M, True)
    fj = jax.jit(
        lambda s: forward_backward(s, *args)["loglik"], static_argnums=(2, 5, 6)
    )
    assert np.isfinite(fj(s)).all()
    fj_grad = jax.jit(jax.grad(fj))
    assert np.isfinite(fj_grad(s)).all()

    from scipy.optimize import check_grad

    cg = check_grad(
        fj,
        fj_grad,
        s,
        epsilon=1e-6,
    )
    assert cg < 1


@pytest.fixture
def M():
    return 100


def test_trans_infinite():
    T = trans_infinite(0.01, 10, 1.0 * jnp.arange(101), jnp.linspace(0, 1e8, 101))


def test_trans_diffusion():
    f = trans_diffusion(0.02, 0.5, 0.01, 1e1 * np.ones(1000), 10, np.eye(11)[4])
    breakpoint()


def test_exact_trans_table1():
    # fix alpha=500  via alpha = tau / (2 eps ** 2)
    alpha = 500
    eps = 1.0
    tau = alpha * 2 * eps ** 2


def test_exact_trans(M):
    d1 = jnp.linspace(0, 2 * M, M)
    d2 = jnp.arange(M)
    s = 0.1
    h = 0.5
    log_T = log_trans_exact(s, h, d1, d2)
    assert log_T.shape == (M, M)
    log_rowsums = logsumexp(log_T, 1)
    np.testing.assert_allclose(0.0, log_rowsums, atol=1e-4)


def test_discretized_trans(M):
    d1 = jnp.arange(M)
    d2 = jnp.linspace(0, 1e5, M)
    s = 0.1
    h = 0.5
    log_T = log_trans_discr(s, h, d1, d2)
    assert log_T.shape == (M, M)
    log_rowsums = logsumexp(log_T, 1)
    np.testing.assert_allclose(0.0, log_rowsums, atol=1e-4)


def test_trans_exp(M):
    d1 = jnp.arange(M)
    d2 = jnp.linspace(0, 1000 * M, M)
    s = 0.1
    h = 0.5
    log_T = log_trans_discr(s, h, d1, d2)
    log_T10 = log_matpow(log_T, 50)
    for A in log_T, log_T10:
        assert A.shape == (M, M)
        log_rowsums = logsumexp(A, 1)
        np.testing.assert_allclose(0.0, log_rowsums, atol=1e-4)
