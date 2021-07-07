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


def compute_rmse(x, y, truth):
    """
    x: vector of T times ranging from N to 0 generations before present
    y: vector of T-1 selection coefficients
    truth: vector N selection values

    Computes the rmse over generations of the selection coefficient
    Treats the selection coefficient as piecewise constant
    i.e. y[0] is the selection coecifficient from T[0] to T[1] etc.. .
    """
    truth = np.array(truth)
    x = [
        max(x) - y for y in x
    ]  # convert from generations bp to generations since start.
    s_est = np.zeros(len(truth))
    for i, s in enumerate(y):
        s_est[x[i] : x[i + 1]] = s
    rmse = np.sqrt(np.average((s_est - truth) ** 2))
    return rmse


def bias_variance(x, y, truth):
    """
    squared bias/variance of estimator
    x: vector of T times ranging from N to 0 generations before present
    y: matrix of K x T-1 selection coefficients (K replicates)
    truth: vector N selection values
    """
    mean_y = np.mean(y, axis=0)
    assert len(set(x)) == 1  # Check all times are the same
    x = x[0]
    x = [
        max(x) - z for z in x
    ]  # convert from generations bp to generations since start.
    s_est = np.zeros((len(y), len(truth)))
    s_avg = np.zeros(len(truth))
    y = np.array(y)
    for i, av in enumerate(mean_y):
        s_est[:, x[i] : x[i + 1]] = np.repeat(y[:, i, None], x[i + 1] - x[i], axis=1)
        s_avg[x[i] : x[i + 1]] = av

    rmbias = np.sqrt(np.average((s_avg - truth) ** 2))
    rmvar = np.sqrt(np.average((s_est - s_avg) ** 2))
    rmse = np.sqrt(np.average((s_est - truth) ** 2))

    return (rmbias, rmvar, rmse)
