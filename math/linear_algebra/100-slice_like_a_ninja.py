#!/usr/bin/env python3

import numpy as np

def np_slice(matrix, axes={}):
    for key, value in axes:
        if key == 0:
            return matrix[: ]
