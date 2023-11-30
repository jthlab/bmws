import numpy as np
import pytest
from scipy.stats import binom

from bmws.common import binom_logpmf, binom_pmf


@pytest.fixture(params=range(10))
def rng(request):
    return np.random.default_rng(request.param)


def test_binom_logpmf(rng: np.random.Generator):
    n = rng.integers(low=0, high=1000)
    k = rng.integers(low=0, high=n)
    p = rng.uniform()
    np.testing.assert_allclose(binom_logpmf(k, n, p), binom.logpmf(k, n, p), rtol=1e-4)


def test_binom_pmf_edge_cases():
    n = 10
    assert binom_pmf(0, n, 0.0) == binom_pmf(n, n, 1.0) == 1.0
    for k in range(1, n + 1):
        assert binom_pmf(k, n, 0.0) == 0.0
    for k in range(n):
        assert binom_pmf(k, n, 1.0) == 0.0


def test_binom_pmf_broadcast():
    p = np.linspace(0, 1, 10)
    k = np.arange(21)
    Ne = 20
    b = binom_pmf(k[None], Ne, p[:, None])
    assert np.isfinite(b).all()
    np.testing.assert_allclose(b.sum(1), 1.0)
