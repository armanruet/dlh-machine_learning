#!/usr/bin/env python3
"""kmean func"""
import numpy as np


def initialize(X, k):
    """def the func"""
    if not isinstance(k, int) or k <= 0:
        return None
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    n, d = X.shape
    low = X.min(axis=0)
    high = X.max(axis=0)

    return np.random.uniform(low, high, size=(k, d))


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    for i in range(iterations):
        C_prev = C.copy()

        # assignment step (vectorized, no loop)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)  # (n, k)
        clss = np.argmin(distances, axis=1)  # (n,)

        # update step (loop over k clusters)
        for j in range(k):
            points_in_cluster = X[clss == j]
            if points_in_cluster.shape[0] == 0:
                C[j] = initialize(X, 1)
            else:
                C[j] = points_in_cluster.mean(axis=0)

        # convergence check
        if np.array_equal(C, C_prev):
            return C, clss

    # final assignment reflecting the last updated C
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
    clss = np.argmin(distances, axis=1)

    return C, clss
