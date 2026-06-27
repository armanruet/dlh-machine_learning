#!/usr/bin/env python3
"""Creating MultiNOrmal"""


import numpy as np


class MultiNormal:
    """Represents a multivariate normal distribution."""

    def __init__(self, data):
        """Defining the function"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True).astype(float)

        centered = data - self.mean
        self.cov = (centered @ centered.T) / (n - 1)
        self.cov = self.cov.astype(float)

    def pdf(self, x):
        """defining PDF"""
        d = self.mean.shape[0]

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        diff = x - self.mean

        exponent = -0.5 * (diff.T @ inv @ diff)
        norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * det)

        return float(norm_const * np.exp(exponent))
