#!/usr/bin/env python3
"""Creating transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """defining the function"""
    return [row for row in zip(*matrix)]
