import dataclasses
import functools
from dataclasses import dataclass
from functools import partial
from typing import List, Callable, Tuple
import numpy as np

from jax import numpy as jnp
from jax._src.scipy.special import gammaincc, gammaln, xlogy, xlog1py
import jax
from jax.experimental.host_callback import id_print
from jax.tree_util import register_pytree_node


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
    r"P(X \le k) for X ~ Poisson(mu)"
    # prevent evaluation of gammainc(x, 0) which causes NaNs, thus making it hard
    # to debug other unrelated NaNs.
    mu0 = mu == 0.0
    mu1 = jnp.where(mu0, 1.0, mu)
    return jnp.where(mu0, k >= 0, gammaincc(jnp.floor(k + 1), mu1))


def binom_logpmf(k, n, p):
    ret = (
        gammaln(1 + n)
        - gammaln(1 + k)
        - gammaln(n - k + 1)
        + xlogy(k, p)
        + xlog1py(n - k, -p)
    )
    # ret = id_print(ret, what="binom_pmf")
    return ret


def binom_pmf(k, n, p):
    return jnp.exp(binom_logpmf(k, n, p))


def binom_logcdf_cp(k, n, p):
    k1 = jnp.where(k == n, 0, k)
    a = 1 / 9 / (n - k1)
    b = 1 / 9 / (k1 + 1)
    r = (k1 + 1) * (1 - p) / (n - k1) / p
    c = (1 - b) * r ** (1.0 / 3)
    mu = 1.0 - a
    sigma = jnp.sqrt(b * r ** (2.0 / 3) + a)
    z = (c - mu) / sigma

    # lots of problems with gradient NaNs when working in log probability space:
    # we have to be extra careful to never take log(x) when x=0 is a computed
    # quantity. thus, many contortions using the where-where trick:
    large = z > 10
    r1 = jnp.where(large, 0.0, jax.scipy.stats.norm.logcdf(jnp.where(large, 0.0, z)))
    small = z < 10
    r2 = jnp.where(
        small, -jnp.inf, jax.scipy.stats.norm.logcdf(jnp.where(small, 0.0, z))
    )

    # r1 = jax.scipy.stats.norm.logcdf((c - mu) / sigma)
    # r2 = id_print(r2, what="r2")
    # # fix up edge cases
    r3 = jnp.where(k == n, 0.0, r1)
    r4 = jnp.where(k == 0, xlog1py(n, -p), r3)
    return r4


def safe_logdiff(A, axis):
    "log(diff(A, axis)) in a way that does not leave NaNs in gradients"
    # (the problem being when A[i]=A[i+1]
    D = jnp.diff(A, axis=axis)
    E = jnp.where(D == 0.0, 1.0, D)
    return jnp.where(E, -jnp.inf, jnp.log(E))


@dataclass(frozen=True)
class Discretization:
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
        mid1 = jnp.geomspace(d, Ne * 0.5, M)
        mid = jnp.concatenate([mid1, (Ne - mid1[:-1])[::-1]])
        high = 1 + Ne + jnp.arange(-d, 0)
        return cls(low, mid, high, Ne)


register_pytree_node(
    Discretization,
    lambda d: (dataclasses.astuple(d), None),  # tell JAX what are the children nodes
    lambda aux_data, children: Discretization(
        *children
    ),  # tell JAX how to pack back into a RegisteredSpecial
)


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


def _scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    keys = list(xs.keys())
    n = len(xs[keys[0]])
    for i in range(n):
        d = {}
        for k in keys:
            if isinstance(xs[k], Discretization):
                d[k] = Discretization(xs[k].low[i], xs[k].mid[i], xs[k].high[i])
            else:
                d[k] = xs[k][i]
        carry, y = f(carry, d)
        ys.append(y)
    return carry, jnp.stack(ys)


def piecewise_safe(
    interval_fns: List[Tuple[Tuple, Callable]], safe_val: float = 1.0
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
    intervals, fns = zip(*interval_fns)

    def ret(x):
        safe = [(a <= x) & (x < b) for a, b in intervals]

        def _g(accum, tup):
            s, f = tup
            x_s = jnp.where(s, x, safe_val)
            return jnp.where(s, f(x_s), accum)

        return functools.reduce(_g, zip(safe, fns), jnp.full_like(x, safe_val))

    return ret


def logexpm1(a):
    # a = id_print(a, what="a")
    # several cases to consider for numerical stability
    f_safe = piecewise_safe(
        [
            ((-np.inf, 1e-20), lambda _: -np.inf),
            ((1e-20, 1e-8), lambda x: jnp.log(x) - x / 2.0),
            ((1e-8, np.log(2.0)), lambda x: jnp.log(-jnp.expm1(-x))),
            ((np.log(2.0), np.inf), lambda x: jnp.log1p(-jnp.exp(-x))),
        ]
    )
    return f_safe(a)


def safe_lse(A, axis):
    "safe logsumexp (don't put NaN in gradient)"
    M = jnp.max(A, axis=axis, keepdims=True)
    M = jnp.where(jnp.isfinite(M), M, 0.0)
    Ap = A - M
    Ap1 = jnp.exp(jnp.where(jnp.isneginf(Ap), 0.0, Ap))
    sumexp = jnp.where(jnp.isneginf(Ap), 0.0, Ap1).sum(axis)
    return A.max(axis=axis) + jnp.where(
        sumexp == 0.0, -jnp.inf, jnp.log(jnp.where(sumexp == 0.0, 1.0, sumexp))
    )


def log_matmul(log_A, log_B):
    # log_A, log_B = id_print((log_A, log_B), what="log_matmul")

    def log_mvp(log_A, log_v):
        # log_A, log_v = id_print((log_A, log_v), what="log_mvp")
        # return jax.scipy.special.logsumexp(log_A + log_v[None, :], axis=1)
        return safe_lse(log_A + log_v[None, :], axis=1)

    return jax.vmap(log_mvp, in_axes=(None, 1), out_axes=1)(log_A, log_B)


def log_matpow(log_A, n):
    if n == 0:
        return jnp.log(jnp.eye(log_A.shape[0]))
    P = log_matpow(log_A, n // 2)
    U = log_matmul(P, P)
    if n % 2 == 0:
        return U
    return log_matmul(U, log_A)


def log_matpow_ub(log_A, n, ub):
    def _f(d, _):
        d1 = jax.lax.cond(
            d["n"] % 2 == 1,
            lambda e: e | {"pow": log_matmul(e["pow"], e["x"])},
            lambda e: e,
            d,
        )
        d1["n"] >>= 1
        # d1 = id_print(d1, what="d2")
        d2 = jax.lax.cond(
            d1["n"] > 0,
            lambda e: e | {"x": log_matmul(e["x"], e["x"])},
            lambda e: e,
            d1,
        )
        # d2 = id_print(d2, what="d2")
        return d2, None

    init = {"pow": np.log(np.eye(log_A.shape[0])), "x": log_A, "n": n}
    # init = id_print(init, what="init")
    ret = jax.lax.scan(_f, init, None, length=ub)[0]
    # ret = id_print(ret, what="ret")
    return ret["pow"]


### TESTS
