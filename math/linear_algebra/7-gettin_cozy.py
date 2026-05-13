#!/usr/bin/env python3
"""Concatenate matrices with axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Defining the function"""
    if axis == 0:
        return mat1 + mat2
    if axis == 1:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
