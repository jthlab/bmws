import pytest
import numpy as np


@pytest.fixture
def rng():
    return np.random.default_rng(seed=1)
