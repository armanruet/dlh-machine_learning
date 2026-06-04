#!/usr/bin/env python3
""" a function def summation_i_squared(n): that calculates"""


def summation_i_squared(n):
    """defining the function"""
    if type(n) not in (int, float):
        return None
    else:
        return int(n * (n + 1) * (2 * n + 1) // 6)
