import dataclasses
import functools
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import jax
import numpy as np
from jax import numpy as jnp
from jax._src.scipy.special import gammaln, xlog1py, xlogy


def f_sh(x: float, s: float, h: float) -> float:
    "allele propagation function for general diploid selection model"
    s *= 2.0
    ret = (
        x
        * (1 + s * h + s * (1 - h) * x)
        / (1 + 2 * s * h * x + s * (1 - 2 * h) * x ** 2)
    )
    # ret = id_print(ret, what="f_sh")
    return ret


@jnp.vectorize
def binom_logpmf(k, n, p):
    kbad = (k < 0) | (k > n)
    k1 = jnp.where(kbad, 0, k)
    p01 = (p == 0) | (p == 1)
    p1 = jnp.where(p01, 0.5, p)
    ret = (
        gammaln(1 + n)
        - gammaln(1 + k1)
        - gammaln(n - k1 + 1)
        + xlogy(k, p1)
        + xlog1py(n - k, -p1)
    )
    ret = jnp.where(kbad, -jnp.inf, ret)
    ret = jnp.where((p == 0) & (k == 0), 0.0, ret)
    ret = jnp.where((p == 0) & (k != 0), -jnp.inf, ret)
    ret = jnp.where((p == 1) & (k == n), 0.0, ret)
    ret = jnp.where((p == 1) & (k != n), -jnp.inf, ret)
    return ret


def binom_pmf(k, n, p):
    return jnp.exp(binom_logpmf(k, n, p))


@dataclass(frozen=True)
class PosteriorDecoding:
    """Posterior decoding obtained from hidden Markov model.

    Args:
        gamma: Array of dimension [T, D] giving the posterior at each time and discretization point.
        t: Array of dimension [T] giving time point in generations corresponding to first axis of gamma.
        discretizations: List of length [T] encoding the discretization at each time point.
    """

    gamma: np.ndarray
    t: np.ndarray
    hidden_states: List[np.ndarray]
    Ne: np.ndarray
    # log_T: np.ndarray
    # log_B: np.ndarray

    def draw(self, ax=None, k=10_000, seed: int = 1) -> "matplotlib.Axis":
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()

        rng = np.random.default_rng(seed)
        S = self.sample(k, rng)
        # ax.plot(self.t, S.mean(axis=1))  # means
        ax.boxplot(S.T, positions=self.t, widths=2, sym="")
        return ax

    def sample(self, k, rng=None):
        """Sample points from this posterior according to weights.

        Args:
            k: number of points to sample
            rng: A Numpy random number generator. If None, the default is used.

        Returns:
            The discretization is assumed to represent a hybrid distribution, with atoms at self.low and self.high,
            and self.mid representing intervals. Conditional on an interval being draw, a point is selected uniformly
            at random from within the interval.
        """
        if rng is None:
            rng = np.random.default_rng()
        ret = np.zeros([len(self.t), k])
        for i, (hs, p, Ne) in enumerate(zip(self.hidden_states, self.gamma, self.Ne)):
            n = rng.choice(len(hs), size=k, replace=True, p=p)
            ret[i] = hs[n] / Ne
        return ret


@dataclass(frozen=True)
class Observation:
    """An observation for the selection HMM.

    Args:
        t: number of generations before present when the sample was collected.
        sample_size: number of individuals sampled.
        num_derived: number of derived alleles in the sample, 0 <= num_derived <= sample_size.
        Ne: effective population size at that generation.

    Notes:
        Missing data can be encoded by setting sample_size=0. This can be used to, for example, indicate past changes in
        the effective population size.
    """

    t: int
    sample_size: int
    num_derived: int
    Ne: int

    def __post_init__(self):
        assert all(
            [
                self.t >= 0,
                self.sample_size >= 0,
                self.num_derived >= 0,
                self.num_derived <= self.sample_size,
                self.Ne > 0,
            ]
        )


def piecewise_safe(
    interval_fns: Dict[Tuple[float, float], Union[Callable[[float], float], float]],
    safe_val: float = 1.0,
) -> Callable:
    """Evaluate a piecewise-defined function such that each piece is only evaluated on its corresponding interval.
    Outside of that interval, the function is evaluated at safeval. This is useful for preventing NaN issues,
    particularly in gradients.

    Args:
        interval: List of intervals.
        fns: List of callables, same length as intervals.
        safe_val: A value which is "safe" for each function, e.g. for which the function is numerically
            stable and returns a real number.

    Returns:
        A function which evaluates according to the rules specified above.
    """
    # use the "where-where" trick to restrict evaluation on each piece.
    intervals, fns = zip(*interval_fns.items())

    def ret(x):
        safe = [(a <= x) & (x < b) for a, b in intervals]

        def _g(accum, tup):
            s, f = tup
            x_s = jnp.where(s, x, safe_val)
            if callable(f):
                f_xs = f(x_s)
            else:
                assert isinstance(f, float)
                f_xs = jnp.full_like(x_s, f)
            return jnp.where(s, f_xs, accum)

        return functools.reduce(_g, zip(safe, fns), jnp.full_like(x, safe_val))

    return ret


def matpow_ub(A, n, ub):
    def _f(d, _):
        d1 = jax.lax.cond(
            d["n"] % 2 == 1,
            lambda e: e | {"pow": e["pow"] @ e["x"]},
            lambda e: e,
            d,
        )
        d1["n"] >>= 1
        # d1 = id_print(d1, what="d2")
        d2 = jax.lax.cond(
            d1["n"] > 0,
            lambda e: e | {"x": e["x"] @ e["x"]},
            lambda e: e,
            d1,
        )
        # d2 = id_print(d2, what="d2")
        return d2, None

    init = {"pow": np.eye(A.shape[0]), "x": A, "n": n}
    # init = id_print(init, what="init")
    ret = jax.lax.scan(_f, init, None, length=ub)[0]
    # ret = id_print(ret, what="ret")
    return ret["pow"]
