"""
spg_crossover.py
================
SPG spectral crossover: D_cross and lambda_star.

Used in the mathematical foundation paper (Theorem 2).

Observable    Description
----------    -----------
lambda_star   Crossover eigenvalue where alpha_eff(lambda*) = -2 exactly.
D_cross       |log(lambda2 / lambda_star)| — distance to spectral crossover.
              D_cross = 0: system is at the crossover.
              D_cross > 0: system is in bulk regime (biological signals).

Reference:
    Arteaga Marroquin, E.A. (2026). Two Exact Results on Spectral
    Heterogeneity in Recurrence Networks. Theorem 2.
    doi: https://doi.org/10.5281/zenodo.20712285
"""

import numpy as np


def spectral_crossover(lambda2, B, p, grid=None):
    """
    Compute the spectral crossover D_cross and lambda_star.

    Parameters
    ----------
    lambda2 : float
        Fiedler value from spg().
    B : array-like
        Numerator values (M2 per node, or equivalent).
    p : array-like
        Denominator values (M1 per node, or equivalent).
    grid : array-like, optional
        Eigenvalue grid for the crossover search.
        Default: logspace(-4, 4, 2000).

    Returns
    -------
    dict with keys:
        lambda_star : crossover eigenvalue (argmax of g(lambda)).
        D_cross     : |log(lambda2 / lambda_star)|.
        g           : array, variance of t_i(lambda) over the grid.
        grid        : array, eigenvalue grid used.

    Notes
    -----
    X_i = B_i / p_i
    t_i(lambda) = 1 / (1 + lambda * X_i)
    g(lambda) = Var(t_i(lambda))
    lambda_star = argmax g(lambda)
    D_cross = |log(lambda2 / lambda_star)|
    """
    X = np.asarray(B) / np.asarray(p)

    if grid is None:
        grid = np.logspace(-4, 4, 2000)

    gvals = np.array([np.var(1.0 / (1.0 + lam * X)) for lam in grid])

    idx = np.argmax(gvals)
    lambda_star = grid[idx]
    D_cross = np.abs(np.log(lambda2 / lambda_star))

    return {
        'lambda_star': lambda_star,
        'D_cross':     D_cross,
        'g':           gvals,
        'grid':        grid,
    }


def D_cross_from_spg(spg_result, grid=None):
    """
    Convenience wrapper: compute D_cross directly from spg() output.

    Parameters
    ----------
    spg_result : dict
        Output of spg_core.spg().
    grid : array-like, optional
        Eigenvalue grid (default: logspace(-4, 4, 2000)).

    Returns
    -------
    dict from spectral_crossover(), or None if spg_result is None.
    """
    if spg_result is None:
        return None
    return spectral_crossover(
        lambda2=spg_result['lambda2'],
        B=spg_result['M2'],
        p=spg_result['M1'],
        grid=grid,
    )
