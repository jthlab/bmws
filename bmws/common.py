import functools
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.special import gammaln, xlog1py, xlogy


def f_sh(x: float, s: float, h: float) -> float:
    "allele propagation function for general diploid selection model"
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
    kf = jnp.where(kbad, 0, k)
    p0, p1 = [jnp.isclose(p, x) for x in (0.0, 1.0)]
    pf = jnp.where(p0 | p1, 0.5, p)
    ret = (
        gammaln(1 + n)
        - gammaln(1 + kf)
        - gammaln(n - kf + 1)
        + xlogy(k, pf)
        + xlog1py(n - k, -pf)
    )
    ret = jnp.where(kbad, -jnp.inf, ret)
    ret = jnp.where(p0 & (k == 0), 0.0, ret)
    ret = jnp.where(p0 & (k != 0), -jnp.inf, ret)
    ret = jnp.where(p1 & (k == n), 0.0, ret)
    ret = jnp.where(p1 & (k != n), -jnp.inf, ret)
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
    B: np.ndarray = None
    T: np.ndarray = None

    def draw(self, ax=None, k=10_000, seed: int = 1) -> "matplotlib.Axis":
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()

        rng = np.random.default_rng(seed)
        S = self.sample(k, rng)
        # ax.plot(self.t, S.mean(axis=1))  # means
        ax.boxplot(S.T, positions=self.t, widths=2, sym="")
        return ax

    def mode(self):
        """Return the posterior mean frequency

        Args:
            None

        Returns:
            vector of posterior mode frequency
        """
        md = np.argmax(self.gamma, axis=1)
        ret = np.zeros([len(self.t)])
        for i, (hs, best, Ne) in enumerate(zip(self.hidden_states, md, self.Ne)):
            ret[i] = hs[best] / Ne

        return ret

    def mean(self):
        """Return the posterior mean frequency

        Args:
            None

        Returns:
            mean: vector of posterior mean frequency
            time points
        """
        ret = np.zeros([len(self.t)])
        for i, (hs, gm, Ne) in enumerate(zip(self.hidden_states, self.gamma, self.Ne)):
            ret[i] = np.sum(gm * hs) / Ne

        return ret

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
