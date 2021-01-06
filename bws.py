"beta-with-spikes transition matrix"
import jax
import jax.numpy as jnp
import numpy as np

from common import f_sh, midpoints


def vbeta(a, b, x):
    # integral_0^x x^(a-1) (1-x)^(b-1)
    # if Ne is large and x is not near the boundary then a,b >> 1 and the normal approximation suffices
    loc = a / (a + b)
    scale = jnp.sqrt(a * b / (a + b + 1)) / (a + b)
    return jax.scipy.stats.norm.cdf(x, loc, scale)


vf_sh = jnp.vectorize(f_sh)


def T_bws(grid: np.ndarray, Ne: float, s: float, h: float) -> np.ndarray:
    "transition matrix for beta-with-spikes model"
    # if len(grid) = G + 1, then the returned matrix has shape (G + 2, G + 2)
    # the last two dimensions are the special absorbing states X=0, X=1
    x = midpoints(grid)
    mu = vf_sh(x, s, h)
    sigma2 = mu * (1 - mu) / (2 * Ne)

    # fixation probabilities
    p0 = (1.0 - x) ** (2 * Ne)
    p1 = x ** (2 * Ne)

    # moments conditioned on no fixation
    mu_star = (mu - p1) / (1.0 - p0 - p1)
    sigma2_star = (sigma2 + mu ** 2 - p1) / (1.0 - p0 - p1) - mu_star ** 2
    c = mu_star * (1.0 - mu_star) / sigma2_star - 1.0
    a_star = mu_star * c
    b_star = (1.0 - mu_star) * c

    # transition matrix conditioned on non-fixation
    B = vbeta(a_star[:, None], b_star[:, None], grid[None])
    T0 = jnp.diff(B, axis=1)
    T0 *= (1 - p0 - p1)[:, None]

    # Add in the absorbing states
    T = jnp.concatenate(
        [
            T0,
            p0[
                :,
                None,
            ],
            p1[:, None],
        ],
        axis=1,
    )
    G = T0.shape[0]
    T = jnp.concatenate([T, jnp.eye(G + 2)[-2:]])
    return T
