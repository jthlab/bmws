import numpy as np
import pyximport

pyximport.install()
import itertools as it

import flsa
from scipy.optimize import OptimizeResult


def fusedlasso(fun, x0, args, jac, **kwargs):
    lam_ = kwargs["lam"]
    tol = kwargs.get("tol", 1e-4)
    x = x0
    f_star = fun(x0, *args)
    for i in it.count(start=0):
        step = 1 / (1 + i)
        while True:
            x_prime = flsa.flsa(
                np.asarray(x - step * jac(x, *args), dtype=np.float64), lam_ * step
            )
            if np.any(np.isnan(x_prime)) or np.any(np.isnan(jac(x_prime, *args))):
                step *= 0.1
            else:
                break
        f_prime = fun(x_prime, *args)
        if i > 100:
            break
        if f_star is not None and i > 10:
            eps = (f_star - f_prime) / f_star
            # print("*", i, eps)
            if eps < tol:
                break
        x = x_prime
        f_star = f_prime
    return OptimizeResult(fun=f_star, x=x)
