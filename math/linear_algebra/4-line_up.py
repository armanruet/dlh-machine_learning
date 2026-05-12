#!/usr/bin/env python3
import numpy as np


def add_arrays(arr1, arr2):
    if np.shape(arr1) == np.shape(arr2):
        return np.add(arr1, arr2)
    else:
        return None
