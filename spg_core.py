"""
spg_core.py
===========
Spectral Phase-space Geometry (SPG) — core pipeline.

Computes lambda2 and C_dyn from a 1D signal via delay embedding
and recurrence network analysis.

Reference:
    Arteaga Marroquin, E.A. (2026). Two Exact Results on Spectral
    Heterogeneity in Recurrence Networks.
    doi: https://doi.org/10.5281/zenodo.20712285
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh


def spg(signal, embed_dim=4, delay=1, percentile=20):
    """
    Core SPG pipeline: signal -> recurrence network -> lambda2, C_dyn.

    Parameters
    ----------
    signal : array-like
        1D time series (single channel, single window).
    embed_dim : int
        Phase-space embedding dimension (default: 4).
    delay : int
        Embedding delay in samples (default: 1).
    percentile : float
        Percentile of pairwise distances used as recurrence threshold (default: 20).

    Returns
    -------
    dict with keys:
        lambda2    : Fiedler value (algebraic connectivity).
        C_dyn      : Var(tau_i) = lambda2^2 * Var(T_i)  [Theorem 1].
        M1         : array, (L+)_ii per node.
        M2         : array, (L+^2)_ii per node.
        tau        : array, lambda2 * T_i per node.
        tau_tilde  : array, T_i = M2/M1 per node.
    Returns None if the signal is flat or the network is degenerate.
    """
    signal = np.asarray(signal, dtype=float)

    # 1. Delay embedding
    N = len(signal) - (embed_dim - 1) * delay
    if N < 6:
        return None
    Y = np.array([signal[i:i + embed_dim * delay:delay] for i in range(N)])

    # 2. Recurrence network
    D = squareform(pdist(Y, metric='euclidean'))
    upper = D[np.triu_indices_from(D, k=1)]
    if len(upper) == 0:
        return None
    eps = np.percentile(upper, percentile)

    A = (D < eps).astype(float)
    np.fill_diagonal(A, 0)

    # Connect isolated nodes to nearest neighbour
    deg = A.sum(axis=1)
    for i in np.where(deg == 0)[0]:
        j = np.argsort(D[i])[1]
        A[i, j] = A[j, i] = 1.0

    # 3. Laplacian spectrum
    L = np.diag(A.sum(axis=1)) - A
    eigvals, eigvecs = eigh(L)

    if eigvals[1] < 1e-10:
        return None
    lambda2 = eigvals[1]

    # 4. Spectral moments
    inv1 = 1.0 / eigvals[1:]
    inv2 = 1.0 / (eigvals[1:] ** 2)
    V = eigvecs[:, 1:] ** 2

    M1 = V @ inv1
    M2 = V @ inv2

    # 5. tau_tilde_i = T_i = M2(i)/M1(i)
    #    tau_i = lambda2 * T_i
    #    C_dyn = Var(tau) = lambda2^2 * Var(T)  [Theorem 1, exact]
    tau_tilde = M2 / M1
    tau = lambda2 * tau_tilde
    C_dyn = float(np.var(tau))

    if not np.isfinite(C_dyn) or C_dyn <= 0:
        return None

    return {
        'lambda2':   lambda2,
        'C_dyn':     C_dyn,
        'M1':        M1,
        'M2':        M2,
        'tau':       tau,
        'tau_tilde': tau_tilde,
    }
