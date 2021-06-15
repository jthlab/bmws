from functools import lru_cache, partial
from typing import NamedTuple

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from cvxpylayers.jax import CvxpyLayer
from jax import (
    jit,
    vmap,
    tree_multimap,
)
from jax.experimental.host_callback import id_print

id_print = lambda x, **kwargs: x
from jax.scipy.special import betaln, xlogy, xlog1py, gammaln, logsumexp


def _logbinom(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _wf_trans(s, N, a, b):
    # X` = Y / N where:
    #
    #     Y ~ Binomial(N, p') | 0 < Y < N ,
    #     p' = p + p(1-p)(s/2),
    #     p ~ Beta(a,b)
    #
    # E(Y | 0 < Y < N) = N(p' - p'^n)
    # EX' = Ep'(1-p'^{N-1})
    a, b = id_print((a, b), what="ab")
    EX = (a * (2 + 2 * a + b * (2 + s))) / (2.0 * (a + b) * (1 + a + b))
    # EX = 0.5 * (
    #     (a * (2 + 2 * a + b * (2 + s))) / (a + b) / (1 + a + b)
    #     - (
    #         (2 * (a + b + N) + b * N * s)
    #         * jnp.exp(
    #             gammaln(a + b) + gammaln(a + N) - gammaln(a) - gammaln(1 + a + b + N)
    #         )
    #     )
    # )
    # var(X') = E var(X'|X) + var E(X' | X) = E p'(1-p')/N + var(x + x(1-x)*(s/2))

    # E(X'|X) = p'(1-p'^(N-1))
    # E p'(1-p') / N
    Evar = (
        (a * b * (4 - a * (-2 + s) + b * (2 + s)))
        / (2.0 * (a + b) * (1 + a + b) * (2 + a + b))
        / N
    )
    varE = (
        a
        * b
        * (
            4 * (1 + a + b) * (2 + a + b) * (3 + a + b)
            - 4 * (a - b) * (1 + a + b) * (3 + a + b) * s
            + (a + a ** 3 - a ** 2 * (-2 + b) + b * (1 + b) ** 2 - a * b * (2 + b))
            * s ** 2
        )
    ) / (4.0 * (a + b) ** 2 * (1 + a + b) ** 2 * (2 + a + b) * (3 + a + b))
    Evar, varE = id_print((Evar, varE), what="Evar/VarE")
    var = Evar + varE
    # EX = E(p')
    # var = E(p'(1-p')/N) + var(p) + (s/2)^2 var(p(1-p)) + s * cov(p, p(1-p))
    u = EX * (1 - EX) / var - 1.0
    u = id_print(u, what="u")
    a1 = u * EX
    b1 = u * (1 - EX)
    a1, b1 = id_print((a1, b1), what="a1b1")
    return a1, b1


class BetaMixture(NamedTuple):
    """Mixture of beta pdfs:

    M = len(c) - 1
    p(x) = sum_{i=0}^{M} c[i] x^(a[i] - 1) (1-x)^(b[i] - 1) / beta(a[i], b[i])
    """

    a: np.ndarray
    b: np.ndarray
    log_c: np.ndarray

    @classmethod
    def uniform(cls, M):
        "Return a mixture f_x with M components, such that f_x(x) === 1, x \in [0, 1]."
        # bernstein polynomial basis:
        # 1 = sum_{i=0}^(M) binom(M,i) x^i (1-x)^(M-i)
        #   = sum_{i=1}^(M+1) binom(M,i-1) x^(i-1) (1-x)^(M-i+2-1)
        #   = sum_{i=1}^(M+1) binom(M,i-1) * beta(i,M-i+2) x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        #   = sum_{i=1}^(M+1) (1+M)^-1 x^(i-1) (1-x)^(M-i+2-1) / beta(i, M-i+2)
        i = jnp.arange(1, M + 1, dtype=float)
        log_c = jnp.full(M, -jnp.log(M))
        return cls(a=i, b=M - i + 1, log_c=log_c)

    @property
    def M(self):
        return len(self.log_c) - 1

    @property
    def moments(self):
        "return a matrix A such that A @ self.c = [EX, EX^2] where X ~ self"
        a = self.a
        b = self.b
        EX = a / (a + b)  # EX
        EX2 = a * (a + 1) / (a + b) / (1 + a + b)  # EX2
        return jnp.array([EX, EX2 - EX ** 2])

    def __call__(self, x):
        return np.exp(
            self.log_c
            + xlogy(self.a - 1, x)
            + xlog1py(self.b - 1, -x)
            - betaln(self.a, self.b)
        ).sum()

    def plot(self, K=100) -> None:
        import matplotlib.pyplot as plt

        x = np.linspace(0.0, 1.0, K)
        y = np.vectorize(self)(x)
        plt.plot(x, y)


class SpikedBeta(NamedTuple):
    log_p0: float
    log_p1: float
    f_x: BetaMixture

    def sample_component(self, rng):
        sub1, sub2, sub3 = jax.random.split(rng, 4)
        i = jax.random.categorical(sub1, self.f_x.log_c)
        p = jax.random.beta(sub2, self.f_x.a[i], self.f_x.b[i])
        j = jax.random.categorical(
            sub3, jnp.array([self.log_p0, self.log_p1, self.log_r])
        )
        return jnp.array([0.0, 1.0, p])[j]

    @property
    def log_r(self):
        return jnp.log1p(-jnp.exp(self.log_p0 + self.log_p1))

    @property
    def M(self):
        return self.f_x.M

    def plot(self):
        import matplotlib.pyplot as plt

        self.f_x.plot()
        plt.bar(0.0, self.p0, 0.05, alpha=0.2, color="tab:red")
        plt.bar(1.0, self.p1, 0.05, alpha=0.2, color="tab:red")


# @partial(jit, static_argnums=4)


def transition(f: SpikedBeta, s: float, Ne: float, n: int, d: int):
    """Given a prior distribution on population allele frequency, compute posterior after
    observing data at dt generations in the future"""

    # compute moments at time dt and transition the latent population allele frequencies

    # this call is jitted

    # var(X') = E var(X'|X) + var E(X' | X)
    #         ~= var(X'|X = EX) + var E(X' | X)
    a = f.f_x.a
    b = f.f_x.b
    log_c = f.f_x.log_c
    # s = id_print(s, what="s")
    # probability mass for fixation. p(beta=0) = p0 + \int_0^1 f(x) (1-x)^n, and similarly for p1
    log_p0 = f.log_p0 + logsumexp(
        log_c + gammaln(a + b) + gammaln(b + Ne) - gammaln(b) - gammaln(a + b + Ne)
    )
    log_p1 = f.log_p1 + logsumexp(
        log_c + gammaln(a + b) + gammaln(a + Ne) - gammaln(a) - gammaln(a + b + Ne)
    )
    # new parameters of each mixture component after w-f sampling
    a1, b1 = _wf_trans(s, Ne, a, b)
    # now model binomial sampling
    # p(d_k|d_{k-1},...,d_1) = \int p(d_k|x_k) p(x_k | d_k-1,..,d1)
    # probability of data arising from each mixing component
    # a1, b1 = id_print((a1, b1), what="wf_trans")
    ret = _binom_sampling(n, d, a1, b1, log_c, log_p0, log_p1)
    ret = id_print(ret, what="ret")
    return ret


def _binom_sampling(n, d, a, b, log_c, log_p0, log_p1):
    log_r = jnp.log1p(-jnp.exp(log_p0 + log_p1))
    log_r, _, _ = id_print((log_r, log_p0, log_p1), what="r/p0/p1")
    a1 = a + d
    b1 = b + n - d
    # probability of the data given each mixing component
    log_p_mix = betaln(a1, b1) - betaln(a, b) + _logbinom(n, d)
    log_p_mix = id_print(log_p_mix, what="p_mix")
    log_p01 = log_c + log_p_mix
    lp0 = log_p0 + jnp.log(d == 0)
    lp1 = log_p1 + jnp.log(d == n)
    ll = logsumexp(jnp.concatenate([lp0[None], log_r + log_p01, lp1[None]]))
    # p(p=1|d) = p(d|p=1)*p01 / pd
    log_p0 = lp0 - ll
    log_p1 = lp1 - ll
    # update mixing coeffs
    # p(c | data) = p(data | c) * p(c) / p(data)
    log_c1 = log_r + log_p01
    log_c1 -= logsumexp(log_c1)
    # posterior after binomial sampling --
    beta = SpikedBeta(log_p0, log_p1, BetaMixture(a1, b1, log_c1))
    # beta = id_print(beta, what="beta")
    return beta, ll


# @partial(jit, static_argnums=3)
def forward(s, Ne, obs, M):
    """
    s: [T - 1] selection coefficient at each time point
    Ne: [T - 1] diploid effective population size at each time point
    obs [T, 2]: (sample size, # derived alleles) observed at each time point
    times: [T] number of generations before present when each observation was sampled, sorted descending.
    """
    n, d = obs.T

    def _f(fX, d):
        beta, ll = transition(fX, **d)
        return beta, (beta, ll)

    # uniform
    f_x = BetaMixture.uniform(M)

    beta0, ll0 = _binom_sampling(
        n[-1], d[-1], f_x.a, f_x.b, f_x.log_c, -jnp.inf, -jnp.inf
    )

    # lls = []
    # betas = []
    # f_x = beta0
    # for i in range(1, 1 + len(Ne)):
    #     f_x, (_, ll) = _f(
    #         f_x, {"Ne": Ne[-i], "n": n[-i - 1], "d": d[-i - 1], "s": s[-i]}
    #     )
    #     betas.append(f_x)
    #     lls.append(ll)

    _, (betas, lls) = jax.lax.scan(
        _f, beta0, {"Ne": Ne, "n": n[:-1], "d": d[:-1], "s": s}, reverse=True
    )

    return (betas, beta0), jnp.concatenate([jnp.array(lls), ll0[None]])


def _tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.
    """
    from jax.tree_util import tree_flatten

    leaves, treedef = tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


def loglik(s, Ne, obs, M=100):
    betas, lls = forward(s, Ne, obs, M)
    return lls.sum()


@partial(jit, static_argnums=4)
def _sample_path(s, Ne, obs, rng, M):
    "draw k samples from the posterior allele frequency distribution"

    def _sample_spikebeta(beta: SpikedBeta, rng, cond):
        # -1, M represent the special fixed states 0/1
        log_p0p = beta.log_p0 + jnp.log(cond != -1)
        log_p1p = beta.log_p1 + jnp.log(cond != -2)
        log_p = jnp.concatenate(
            [log_p0p[None], log_p1p[None], beta.log_r + beta.f_x.log_c]
        )
        sub1, sub2 = jax.random.split(rng)
        s = jax.random.categorical(sub1, log_p) - 2
        x = jnp.where(
            s < 0,
            jnp.take(jnp.array([0.0, 1.0]), 2 + s),
            jax.random.beta(sub2, beta.f_x.a[s], beta.f_x.b[s]),
        )
        return (s, x)

    (betas, beta0), _ = forward(s, Ne, obs, M)
    rng, sub = jax.random.split(rng)
    s0, x0 = _sample_spikebeta(beta0, sub, True)

    def _f(tup, beta):
        rng, s1 = tup
        rng, sub1, sub2 = jax.random.split(rng, 3)
        s, x = _sample_spikebeta(beta, sub1, s1)
        return (rng, s), x

    _, xs = jax.lax.scan(_f, (rng, s0), betas, reverse=True)
    return jnp.concatenate([xs, x0[None]])


def sample_paths(s, Ne, obs, k, rng=jax.random.PRNGKey(1), M=10):
    paths = []
    for _ in range(k):
        rng, sub = jax.random.split(rng)
        paths.append(_sample_path(s, Ne, obs, sub, M))
    return np.array(paths)
