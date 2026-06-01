#!/usr/bin/env python3
""" a function def summation_i_squared(n): that calculates"""


def summation_i_squared(n):
    sum = 0
    for i in range(n+1):
        sum += i**2
    return sum
