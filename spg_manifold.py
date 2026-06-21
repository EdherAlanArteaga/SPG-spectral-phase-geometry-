"""
spg_manifold.py
===============
SPG manifold position: (lambda2(t), C_dyn(t)) -> d_eff, CI.

Used in Paper 2: "A Geometric Manifold of Brain State Organization".

Observable    Description
---------     -----------
d_eff         Effective dimensionality = tr(Sigma)^2 / tr(Sigma^2)
              where Sigma = Cov(lambda2_series, C_dyn_series)
CI            Crystallization Index = |Spearman(lambda2, C_dyn)|
              CI -> 1: crystallized (coma, deep sleep)
              CI -> 0: exploratory (wakefulness)
"""

import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr
from spg_core import spg


def manifold_position(lambda2_series, C_dyn_series):
    """
    Compute manifold coordinates from time series of SPG observables.

    Parameters
    ----------
    lambda2_series : array-like, shape (n_windows,)
        Fiedler values across sliding windows.
    C_dyn_series : array-like, shape (n_windows,)
        C_dyn values across sliding windows.

    Returns
    -------
    dict with keys:
        d_eff  : effective dimensionality, tr(Sigma)^2 / tr(Sigma^2).
        CI     : Crystallization Index, |Spearman(lambda2, C_dyn)|.
        Sigma  : 2x2 covariance matrix of (lambda2, C_dyn).
    """
    X = np.column_stack([lambda2_series, C_dyn_series])
    Sigma = np.cov(X.T)

    tr1 = np.trace(Sigma)
    tr2 = np.trace(Sigma @ Sigma)

    d_eff = (tr1 ** 2) / tr2 if tr2 > 0 else np.nan

    rho, _ = spearmanr(lambda2_series, C_dyn_series)
    CI = abs(rho)

    return {
        'd_eff': d_eff,
        'CI':    CI,
        'Sigma': Sigma,
    }


def eeg_to_manifold(signal, fs, window_sec=4, step_sec=2,
                    embed_dim=4, delay=1, percentile=20, max_windows=40):
    """
    Complete pipeline: EEG signal -> manifold position (d_eff, CI).

    Parameters
    ----------
    signal : array-like
        1D EEG time series (single channel).
    fs : float
        Sampling frequency in Hz.
    window_sec : float
        Window length in seconds (default: 4).
    step_sec : float
        Step size in seconds (default: 2).
    embed_dim : int
        Embedding dimension (default: 4).
    delay : int
        Embedding delay in samples (default: 1).
    percentile : float
        Recurrence threshold percentile (default: 20).
    max_windows : int
        Maximum number of windows to process (default: 40).

    Returns
    -------
    dict with keys:
        lambda2_series : array of Fiedler values per window.
        C_dyn_series   : array of C_dyn values per window.
        d_eff          : manifold position.
        CI             : Crystallization Index.
        Sigma          : 2x2 covariance matrix.
    Returns None if fewer than 6 valid windows.
    """
    win = int(window_sec * fs)
    step = int(step_sec * fs)

    lambda2_ts, cdyn_ts = [], []

    for n, start in enumerate(range(0, len(signal) - win + 1, step)):
        if n >= max_windows:
            break
        out = spg(signal[start:start + win],
                  embed_dim=embed_dim, delay=delay, percentile=percentile)
        if out is not None:
            lambda2_ts.append(out['lambda2'])
            cdyn_ts.append(out['C_dyn'])

    if len(lambda2_ts) < 6:
        return None

    lambda2_ts = np.array(lambda2_ts)
    cdyn_ts = np.array(cdyn_ts)

    manifold = manifold_position(lambda2_ts, cdyn_ts)

    return {
        'lambda2_series': lambda2_ts,
        'C_dyn_series':   cdyn_ts,
        'd_eff':          manifold['d_eff'],
        'CI':             manifold['CI'],
        'Sigma':          manifold['Sigma'],
    }
