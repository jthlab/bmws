mdls = [
    {"s": [0.02] * 50 + [-0.02] * 50, "h": [0.5] * 100, "f0": 0.1},
    {"s": [0.02] * 100 + [0.0] * 50 + [-0.02] * 50, "h": [0.5] * 200, "f0": 0.1},
    {"s": (([0.02] * 40 + [-0.02] * 40) * 3)[:200], "h": [0.5] * 200, "f0": 0.5},
]

import numpy as np
import pytest

from infer import sim_and_fit


@pytest.mark.parametrize("Ne", 10 ** np.arange(0, 10))
def test_extreme_Ne(Ne):
    res = sim_and_fit(
        mdls[0], seed=1, lam_=1.0, s_mode=True, fixed_val=1 / 2, ell1=False, Ne=Ne
    )
    assert np.isfinite(res["x"]).all()
