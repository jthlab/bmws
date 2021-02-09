import numpy as np
import pytest
from scipy.stats import binom

from common import binom_logpmf


@pytest.fixture(params=range(10))
def rng(request):
    return np.random.default_rng(request.param)


def test_binom_logpmf(rng: np.random.Generator):
    n = rng.integers(low=0, high=1000)
    k = rng.integers(low=0, high=n)
    p = rng.uniform()
    np.testing.assert_allclose(binom_logpmf(k, n, p), binom.logpmf(k, n, p), rtol=1e-4)
