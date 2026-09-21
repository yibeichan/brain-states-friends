"""Stationary distribution of a Markov chain restricted to a state subset.

Mirrors the computation in the main pipeline's 06b_transition_structure
(compute_mfpt): take the transition sub-matrix over the requested states,
renormalize rows if they deviate from one, and return the left eigenvector
for the eigenvalue nearest 1 as a probability vector. Frozen snapshot for
the supplements branch; kept dependency-free (numpy only).
"""
import numpy as np


def stationary_distribution(transmat, states):
    """Return pi over `states` (ascending order) with pi @ P = pi.

    Parameters
    ----------
    transmat : (K, K) array
        Full model transition matrix.
    states : iterable of int
        State indices forming a closed (strongly connected) subset.
    """
    idx = np.asarray(sorted(int(s) for s in states), dtype=int)
    P = np.asarray(transmat, dtype=float)[np.ix_(idx, idx)].copy()
    row_sums = P.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("transition sub-matrix has an all-zero row; subset is not closed")
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        P = P / row_sums[:, np.newaxis]
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = np.real(eigenvectors[:, k])
    if pi.sum() < 0:
        pi = -pi
    pi = pi / pi.sum()
    pi = np.maximum(pi, 1e-15)
    return pi
