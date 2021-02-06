import jax
import numpy as np
import pytest
from jax import numpy as jnp
from scipy.stats import binom
import scipy

from common import (
    binom_logpmf,
    binom_logcdf_cp,
    log_matmul,
    log_matpow,
    log_matpow_ub,
    safe_lse,
    logexpm1,
)


@pytest.fixture(params=range(10))
def rng(request):
    return np.random.default_rng(request.param)


def test_binom_logpmf(rng: np.random.Generator):
    n = rng.integers(low=0, high=1000)
    k = rng.integers(low=0, high=n)
    p = rng.uniform()
    np.testing.assert_allclose(binom_logpmf(k, n, p), binom.logpmf(k, n, p), rtol=1e-4)


def test_binom_cdf(rng):
    n = rng.integers(low=0, high=1000)
    k = rng.integers(low=0, high=n)
    p = rng.uniform()
    np.testing.assert_allclose(
        np.exp(binom_logcdf_cp(k, n, p)),
        binom.cdf(k, n, p),
        atol=0.007 / np.sqrt(n * p * (1 - p)),
    )  # this is the theoretical ub for the camp paulson approximation


def test_log_matmul():
    A = np.random.rand(7, 8) ** 2
    B = np.random.rand(8, 7) ** 2
    C = A @ B
    log_C = log_matmul(np.log(A), np.log(B))
    np.testing.assert_allclose(C, np.exp(log_C), rtol=1e-4)


def test_log_matmul_eye():
    log_A = jnp.log(np.random.rand(8, 8) ** 2)
    B = jnp.eye(log_A.shape[0])
    log_C = log_matmul(log_A, np.log(B))
    np.testing.assert_allclose(log_A, log_C)


def test_log_matpow2():
    A = np.random.rand(8, 8) ** 2
    B = np.linalg.matrix_power(A, 2)
    log_B = log_matpow(np.log(A), 2)
    np.testing.assert_allclose(B, np.exp(log_B), rtol=1e-4)


def test_log_matpow3():
    A = np.random.rand(8, 8) ** 2
    B = np.linalg.matrix_power(A, 3)
    log_B = log_matpow(np.log(A), 3)
    np.testing.assert_allclose(B, np.exp(log_B), rtol=1e-4)


def test_log_matpow5():
    A = np.random.rand(8, 8) ** 2
    B = np.linalg.matrix_power(A, 5)
    log_B = log_matpow(np.log(A), 5)
    np.testing.assert_allclose(B, np.exp(log_B), rtol=1e-4)


def test_log_matmul_grad():
    log_A = np.random.exponential(size=(8, 8))
    g = jax.grad(lambda log_A: log_matmul(log_A, log_A).sum())(log_A)
    assert np.isfinite(g).all()


def test_log_matpow_grad():
    log_A = np.random.exponential(size=(8, 8))
    g = jax.grad(lambda log_A: log_matpow(log_A, 10).sum())(log_A)
    assert np.isfinite(g).all()


@pytest.mark.parametrize("n", range(1, 11))
def test_log_matpow_ub(rng: np.random.Generator, n):
    log_A = rng.exponential(size=(n, n))
    log_B1 = log_matpow(log_A, n)
    log_B2 = log_matpow_ub(log_A, n, 1 + int(np.log2(max(1, n))))
    np.testing.assert_allclose(log_B1, log_B2, rtol=1e-4)


def test_grad_matmul():
    log_A = np.random.exponential(size=(2, 2))
    g = jax.grad(lambda A, k: log_matpow(A, k).sum())(log_A, 10)
    assert np.isfinite(g).all()


def test_safe_lse(rng: np.random.Generator):
    A = rng.random((10, 10))
    for ax in [None, 0, 1]:
        np.testing.assert_allclose(
            scipy.special.logsumexp(A, axis=ax), safe_lse(A, axis=ax), rtol=1e-4
        )


def test_safe_lse_inf():
    A = np.full((4, 4), -np.inf)
    for ax in [None, 0, 1]:
        assert np.isneginf(safe_lse(A, ax)).all()


def test_logexpm1_asymp():
    a = 1e-10
    np.testing.assert_allclose(logexpm1(a), -23.025850929890456840)  # mathematica


def test_logexpm1_inf():
    a = 0.0
    assert np.isneginf(logexpm1(a))


def test_logexpm1_rand(rng):
    for x in rng.exponential(size=10):
        np.testing.assert_allclose(np.log(-np.expm1(-x)), logexpm1(x), rtol=1e-4)
