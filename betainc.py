import jax

jax.config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln, xlog1py, xlogy

# https://github.com/canerturkmen/betaincder/blob/master/betaincder/c/beta.c




def dj(a, b, x, m):
    m_even = m // 2
    m_odd = (m - 1) // 2
    return jnp.where(
        m % 2 == 0,
        m_even * (b - m_even) * x / (a + 2 * m_even - 1) / (a + 2 * m_even),
        -(a + m_odd) * (a + b + m_odd) * x / (a + 2 * m_odd) / (a + 2 * m_odd + 1),
    )


def betaln(a, b):
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def jax_betainc0(a, b, x):
    M = 10
    r, _ = jax.lax.scan(
        lambda r, m: (dj(a, b, x, m) / (1 + r), None),
        dj(a, b, x, M),
        jnp.arange(M, 0, -1),
    )
    r = 1 / (1 + r)
    x01 = (x == 0.0) | (x == 1.0)
    xc = jnp.where(x01, 0.5, x)  # avoid taking log of 0
    y = xlogy(a, xc) + xlog1py(b, -xc) - jnp.log(a) - betaln(a, b)
    ret = r * jnp.exp(y)
    ret = jnp.where(x == 0.0, 0.0, ret)
    ret = jnp.where(x == 1.0, 1.0, ret)
    return ret


def betainc(a, b, x):
    cond = (x > (a + 1) / (a + b + 2)) | ((1 - x) < (b + 1) / (a + b + 2))
    return jax.lax.cond(
        cond,
        lambda _: 1 - jax_betainc0(b, a, 1 - x),
        lambda _: jax_betainc0(a, b, x),
        None,
    )


import logging

logging.getLogger("absl").setLevel(logging.DEBUG)
import numpy as np
import pytest
import scipy.special
from scipy.optimize import check_grad


@pytest.fixture(scope="module", params=np.arange(10))
def abx(request):
    rng = np.random.default_rng(request.param)
    return (rng.exponential(), rng.exponential(), rng.random())


def test_vs_scipy(abx):
    u = scipy.special.betainc(*abx)
    v = betainc(*abx)
    assert abs(u - v) < 1e-4


def test_betainc_grad(abx):
    def f(u):
        return betainc(*u)

    g = jax.grad(f)
    jg = jax.jit(g)
    for gg in g, jg:
        delta = check_grad(f, gg, abx)
        assert delta < 1e-5, (delta, x, a, b, x < a / (a + b))
