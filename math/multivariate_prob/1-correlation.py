#!/usr/bin/env python3
"""creating mean_cov"""
import numpy as np


def correlation(C):
    """defining the function"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std = np.sqrt(np.diag(C))
    denom = np.outer(std, std)

    corr = np.divide(C, denom, out=np.zeros_like(
        C, dtype=float), where=denom != 0)
    np.fill_diagonal(corr, 1.0)

    return corr
