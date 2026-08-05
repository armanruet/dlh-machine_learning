#!/usr/bin/env python3
"""
Module that calculates the probability density function of a
Gaussian distribution
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian
    distribution

    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    d = m.shape[0]
    if X.shape[1] != d or S.shape != (d, d):
        return None

    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)

        X_centered = X - m
        exponent = -0.5 * np.sum(
            np.dot(X_centered, inv) * X_centered, axis=1
        )

        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)

        P = norm_const * np.exp(exponent)
        P = np.maximum(P, 1e-300)

        return P
    except Exception:
        return None
