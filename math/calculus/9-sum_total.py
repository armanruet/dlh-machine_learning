#!/usr/bin/env python3
""" a function def summation_i_squared(n): that calculates"""

def summation_i_squared(n):
    """defining the function"""
    if not isinstance(n, int) or isinstance(n, bool):
        return None
    else:
        return n * (n + 1) * (2 * n + 1) // 6
