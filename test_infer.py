from common import Observation, Discretization
from estimate import posterior_decoding

mdls = [
    {"s": [0.01] * 100, "h": [0.5] * 100, "f0": 0.1},
    {"s": [0.02] * 50 + [-0.02] * 50, "h": [0.5] * 100, "f0": 0.1},
    {"s": [0.02] * 100 + [0.0] * 50 + [-0.02] * 50, "h": [0.5] * 200, "f0": 0.1},
    {"s": (([0.02] * 40 + [-0.02] * 40) * 3)[:200], "h": [0.5] * 200, "f0": 0.5},
]

import numpy as np
import pytest

from sim import sim_and_fit, sim_wf

import logging

logging.getLogger("absl").setLevel(logging.DEBUG)
logging.getLogger().setLevel(logging.DEBUG)


@pytest.fixture
def rng():
    return np.random.default_rng()


def test_make_discretization_d0():
    mi = Discretization.factory(64, 0, int(1e3))
    assert len(mi.low) == len(mi.low) == 0
    assert len(mi.mid) == 64


def test_basic_scenario():
    res = sim_and_fit(mdls[0], seed=1, lam_=1.0, d=50, M=50)


def test_posterior(rng):
    Ne = int(1e3)
    mdl = mdls[0]
    af = sim_wf(Ne, mdl["s"], mdl["h"], mdl["f0"], rng)
    n = 100
    k = 10
    obs = rng.binomial(n, af[::k])  # sample n haploids every d generations

    data = [
        Observation(t=t, sample_size=n, num_derived=oo, Ne=Ne)
        for t, oo in zip(range(0, len(af), k), obs)
    ]
    pd = posterior_decoding(data, mdl["s"][::k], mdl["h"][::k], 64)
    S = pd.sample(100)
    assert S.shape == (len(data), 100)
    # check that plotting works
    import matplotlib

    matplotlib.use("Agg")
    pd.draw()


def test_huge_Ne():
    import jax

    jax.config.update("jax_disable_jit", 1)
    Ne = int(1e6)
    rng = np.random.default_rng()
    mdl = mdls[1]
    af = sim_wf(Ne, mdl["s"], mdl["h"], mdl["f0"], rng)
    n = 100
    k = 10
    obs = rng.binomial(n, af[::k])  # sample n haploids every d generations

    data = [
        Observation(t=t, sample_size=n, num_derived=oo, Ne=Ne)
        for t, oo in zip(range(0, len(af), k), obs)
    ]

    pd = posterior_decoding(
        data,  # observed data
        mdl["s"][::k],
        mdl["h"][::k],  # parameters for HMM
        64,  # number of discretizations
    )
