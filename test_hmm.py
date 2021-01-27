import jax
import numpy as np

from hmm import forward_backward

# testing

import logging

logging.getLogger("absl").setLevel(logging.DEBUG)


def test_hmm():
    times = tuple(range(0, 200, 10))[::-1]
    T = len(times)
    s = np.zeros_like(times[:-1])
    h = np.zeros_like(s)
    Ne = (100,) * T
    n = np.random.randint(2, 20, size=T)
    derived = np.random.randint(0, n, size=T)
    obs = np.transpose([n, derived])
    d = 10
    M = 100
    gamma = forward_backward(times, s, h, Ne, obs, d, M, False)
    np.testing.assert_allclose(gamma.sum(1), 1.0, atol=1e-4)
    np.testing.assert_allclose(gamma.sum(), T, atol=1e-4)


def test_grad():
    times = tuple(range(0, 200, 75))[::-1]
    T = len(times)
    s = np.zeros(T - 1) + 0.01
    h = np.zeros_like(s) + 0.01
    Ne = (100,) * T
    n = np.random.randint(2, 20, size=T)
    derived = np.random.randint(0, n, size=T)
    obs = np.transpose([n, derived])
    d = 3
    M = 5

    f_grad = jax.grad(forward_backward, argnums=(1, 2))(
        times, s, h, Ne, obs, d, M, True
    )
