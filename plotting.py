# Shared plotting and i/o code
import numpy as np


def plot_summary(ax, x, y, truth=None, q=[0.025, 0.5, 0.975]):
    lower, mid, upper = np.quantile(y, q, axis=0)
    assert len(set(x)) == 1
    x = x[0]
    ax.plot(x, mid, color="tab:blue")
    ax.fill_between(x, lower, upper, alpha=0.2, color="tab:blue")
    if truth is not None:
        x0 = np.arange(len(truth))
        ax.plot(x0, truth, "--", alpha=0.5, color="tab:grey")


def compute_rmse(y, truth):
    """
    y: vector of N selection coefficients
    truth: vector N selection values

    Computes the rmse over generations of the selection coefficient
    Treats the selection coefficient as piecewise constant
    i.e. y[0] is the selection coecifficient from T[0] to T[1] etc.. .
    """
    truth = np.array(truth)
    s_est = np.array(y)
    rmse = np.sqrt(np.average((s_est - truth) ** 2))
    return rmse


def bias_variance(y, truth):
    """
    squared bias/variance of estimator
    x: vector of T times ranging from N to 0 generations before present
    y: matrix of K x T-1 selection coefficients (K replicates)
    truth: vector N selection values
    """
    truth = np.array(truth)
    s_est=np.array(y)
    s_avg = np.mean(y, axis=0)

    rmbias = np.sqrt(np.average((s_avg - truth) ** 2))
    rmvar = np.sqrt(np.average((s_est - s_avg) ** 2))
    rmse = np.sqrt(np.average((s_est - truth) ** 2))
 
    return (rmbias, rmvar, rmse)
