#!/usr/bin/env python3
"""Module that implements forward propagation over a pooling layer."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Parameters:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
                containing the output of the previous layer
        kernel_shape: tuple of (kh, kw) containing the kernel size
        stride: tuple of (sh, sw) containing the strides
        mode: string containing either 'max' or 'avg'

    Returns:
        The output of the pooling layer with shape (m, h_out, w_out, c_prev)
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # 1. Compute output spatial dimensions
    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    # 2. Allocate output volume (channels remain identical to c_prev)
    A = np.zeros((m, h_out, w_out, c_prev))

    # 3. Slide the pooling window across height and width
    for i in range(h_out):
        v_start = i * sh
        v_end = v_start + kh

        for j in range(w_out):
            h_start = j * sw
            h_end = h_start + kw

            # Extract window across all m examples and all c_prev channels
            slice_A = A_prev[:, v_start:v_end, h_start:h_end, :]

            # 4. Perform the selected reduction across spatial axes (1, 2)
            if mode == 'max':
                A[:, i, j, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(slice_A, axis=(1, 2))

    return A
