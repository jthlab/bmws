# cython: boundscheck = False
# cython: language_level = 3

"""Dynamic programming algorithm for the 1d fused lasso problem.
(Jonathan's implementation of Ryan's implementation of Nick Johnson's algorithm)
(see http://www.stat.cmu.edu/~ryantibs/convexopt-F13/homeworks/prox_matlab.cpp)
"""

import numpy as np

cdef extern:
    void prox_dp(int, double*, double, double*) nogil

def flsa(const double[:] y, double lam):
    '''Solve the fused lasso problem

        min (1/2) sum_i (y_i - beta_i) ** 2 + lam_ ||beta||_1

    Args:
        y: observations
        lam_: tuning parameter

    Returns:
        beta: fitted values
    '''
    # this is pretty much a line-by-line translation of the matlab file linked above
    cdef int n = len(y)
    ret = np.zeros(n)
    cdef double[:] beta = ret
    with nogil:
        prox_dp(n, &y[0], lam, &beta[0])
    return ret
