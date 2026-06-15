#!/usr/bin/env python3

"""Likelihood"""
import numpy as np


def fact(n):
    if n == 0:
        return 1
    return n*fact(n-1)


def likelihood(x, n, P):
    """defining the function"""
    if n <= 0 or not isinstance(n, int):
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not ((P >= 0) & (P <= 1)).all():
        raise ValueError("All values in P must be in the range [0, 1]")
    ncx = fact(n)/(fact(n-x)*fact(x))
    return ncx*(P**x)*((1-P)**(n-x))
