#!/usr/bin/env python3
"""Shuffle two matrices using the same random permutation."""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffle the rows of X and Y using the same random permutation.

    """
    m = X.shape[0]

    indices = np.random.permutation(m)

    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    return X_shuffled, Y_shuffled
