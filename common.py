def midpoints(grid):
    return (grid[:-1] + grid[1:]) / 2.0


def f_sh(x: float, s: float, h: float) -> float:
    "allele propagation function for general diploid selection model"
    s *= 2.0
    return (
        x
        * (1 + s * h + s * (1 - h) * x)
        / (1 + 2 * s * h * x + s * (1 - 2 * h) * x ** 2)
    )
