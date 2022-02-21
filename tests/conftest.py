import numpy as np
import pytest


@pytest.fixture
def times():
    return tuple(range(0, 200, 10))[::-1]


@pytest.fixture
def T(times):
    return len(times)


@pytest.fixture
def Ne(T):
    return np.array([100.0] * T)


@pytest.fixture
def s(times):
    return np.zeros(len(times[:-1]))


@pytest.fixture
def obs(times):
    T = len(times)
    n = np.random.randint(2, 20, size=T)
    derived = np.random.randint(0, n, size=T)
    obs = np.transpose([n, derived])
    obs[-1][:] = 10
    return obs
