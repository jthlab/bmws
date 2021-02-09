import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from hmm import forward_backward, trans_exact, trans_infinite

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
    gamma = forward_backward(s, h, times, Ne, obs, M, False)["gamma"]
    np.testing.assert_allclose(gamma.sum(1), 1.0, atol=1e-4)
    np.testing.assert_allclose(gamma.sum(), T, atol=1e-4)


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
    gamma = forward_backward(s, h, times, Ne, obs, M, False)["gamma"]
    np.testing.assert_allclose(gamma.sum(1), 1.0, atol=1e-4)
    np.testing.assert_allclose(gamma.sum(), T, atol=1e-4)


def test_grad_loglik():
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
    fj = jax.jit(lambda s: forward_backward(s, *args)["loglik"])
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
    np.testing.assert_allclose(1.0, T.sum(1), atol=1e-4)


def test_exact_trans(M):
    d1 = jnp.arange(2 * M + 1)
    d2 = jnp.arange(M + 1)
    s = 0.1
    T = trans_exact(s, d1, 2 * M, d2, M)
    assert T.shape == (2 * M + 1, M + 1)
    np.testing.assert_allclose(1.0, T.sum(1), atol=1e-4)
