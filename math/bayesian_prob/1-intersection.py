#!/usr/bin/env python3
"""Likelihood"""
import numpy as np


def fact(n):
    """def fact"""
    if n == 0:
        return 1
    return n*fact(n-1)


def likelihood(x, n, P):
    """defining the function"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not ((P >= 0) & (P <= 1)).all():
        raise ValueError("All values in P must be in the range [0, 1]")
    ncx = fact(n)/(fact(n-x)*fact(x))
    return ncx*(P**x)*((1-P)**(n-x))


def intersection(x, n, P, Pr):
    """defining the function"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not ((P >= 0) & (P <= 1)).all():
        raise ValueError("All values in P must be in the range [0, 1]")
    if P.ndim != Pr.ndim:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    return likelihood(x, n, P)*Pr
