#!/usr/bin/env python3
"""initializes cluster centroids"""
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
