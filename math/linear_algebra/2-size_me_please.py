#!/usr/bin/env python3
"""Shape of a matrix"""


def matrix_shape(matrix):
    """defining the function"""
    shape = []
    x = matrix
    while isinstance(x, list) and len(x) > 0:
        shape.append(len(x))
        x = x[0]   # go one level deeper
    return shape
