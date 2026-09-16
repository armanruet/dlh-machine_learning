#!/usr/bin/env python3
"""Module that implements backward propagation over a pooling layer."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backward propagation over a pooling layer of a neural network.

    Parameters:
        dA: numpy.ndarray of shape (m, h_new, w_new, c)
            containing partial derivatives with respect to pooling output
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
            containing the output of the previous layer
        kernel_shape: tuple of (kh, kw) containing the pooling kernel size
        stride: tuple of (sh, sw) containing the strides
        mode: string containing either 'max' or 'avg'

    Returns:
        dA_prev: partial derivatives with respect to previous layer
                 of shape (m, h_prev, w_prev, c)
    """
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # 1. Initialize downstream gradient accumulator
    dA_prev = np.zeros_like(A_prev)

    # 2. Iterate across output spatial coordinates
    for i in range(h_new):
        v_start = i * sh
        v_end = v_start + kh

        for j in range(w_new):
            h_start = j * sw
            h_end = h_start + kw

            # Current gradient slice with shape (m, 1, 1, c)
            da = dA[:, i:i+1, j:j+1, :]

            if mode == 'max':
                # Slice input window across all batch items and channels
                slice_A = A_prev[:, v_start:v_end, h_start:h_end, :]

                # Boolean mask where each element matches the window maximum
                mask = (slice_A == np.max(slice_A, axis=(1, 2), keepdims=True))

                # Route gradient solely to the maximum coordinate(s)
                dA_prev[:, v_start:v_end, h_start:h_end, :] += mask * da

            elif mode == 'avg':
                # Distribute gradient equally across the kh * kw window
                dA_prev[:, v_start:v_end, h_start:h_end, :] += da / (kh * kw)

            else:
                raise ValueError("mode must be either 'max' or 'avg'")

    return dA_prev
