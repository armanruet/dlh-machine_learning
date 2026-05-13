#!/usr/bin/env python3
"""Concatenating matrices in axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """defining the function"""
    return np.concatenate((mat1, mat2), axis)
