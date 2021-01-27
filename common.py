from dataclasses import dataclass
from typing import NamedTuple, List
import numpy as np

from jax import numpy as jnp
from jax._src.scipy.special import gammaincc, gammaln, xlogy, xlog1py


from jax.experimental.host_callback import id_print


def midpoints(grid):
    return (grid[:-1] + grid[1:]) / 2.0


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


def xform(x):
    # ensure that 1+s, 1+sh > 0
    return jnp.expm1(x)


def poisson_cdf(k, mu):
    "P(X \le k) for X ~ Poisson(mu)"
    ret = gammaincc(jnp.floor(k + 1), mu)
    # ret = id_print(ret, what="poisson_cdf")
    return ret


def binom_pmf(k, n, p):
    ret = jnp.exp(
        gammaln(1 + n)
        - gammaln(1 + k)
        - gammaln(n - k + 1)
        + xlogy(k, p)
        + xlog1py(n - k, -p)
    )
    # ret = id_print(ret, what="binom_pmf")
    return ret


class Discretization(NamedTuple):
    low: jnp.ndarray
    mid: jnp.ndarray
    high: jnp.ndarray
    Ne: int

    def untree(self):
        return list(map(self._make, zip(*self)))

    @property
    def U(self):
        return jnp.concatenate([self.low, self.mid, self.high], axis=self.low.ndim - 1)

    @property
    def p(self):
        return (
            jnp.concatenate(
                [self.low, (self.mid[..., :-1] + self.mid[..., 1:]) / 2.0, self.high],
                axis=self.low.ndim - 1,
            )
            / (1.0 * self.Ne)
        )

    @property
    def M(self):
        return self.p.shape[-1]

    def _slice(self, sl):
        return Discretization(self.low[sl], self.mid[sl], self.high[sl], self.Ne[sl])

    # these methods are useful for slicing vmapped Discretizations which have an added (assumed) 0th axis
    @property
    def head(self):
        return self._slice(slice(None, -1, None))

    @property
    def tail(self):
        return self._slice(slice(1, None, None))

    @classmethod
    def factory(cls, M, d, Ne) -> "Discretization":
        low = jnp.arange(d)
        mid = jnp.linspace(d, Ne - d, M)
        high = 1 + Ne + jnp.arange(-d, 0)
        return cls(low, mid, high, Ne)


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
    discretizations: List[Discretization]

    def draw(self, ax=None, k=10_000) -> "matplotlib.Axis":
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()

        rng = np.random.default_rng(1)
        S = self.sample(k, rng)
        ax.plot(self.t, S.mean(axis=1))  # means
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
        for i, (d, p) in enumerate(zip(self.discretizations, self.gamma)):
            n = rng.choice(len(d.U[:-1]), size=k, replace=True, p=p)
            ret[i] = rng.integers(np.ceil(d.U[n]), np.floor(d.U[n + 1])) / d.Ne
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
