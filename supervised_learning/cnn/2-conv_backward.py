#!/usr/bin/env python3
"""Module that implements backward propagation over a convolutional layer."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs  propagation over a convolutional layer of a neural network.

    Parameters:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
            containing the partial derivatives with respect to the unactivated
            output of the convolutional layer
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution
        b: numpy.ndarray of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution
        padding: string, either "same" or "valid"
        stride: tuple of (sh, sw) containing the strides

    Returns:
        dA_prev: gradient of the cost with respect to the input A_prev
                 (m, h_prev, w_prev, c_prev)
        dW: gradient of the cost with respect to the weights W
            (kh, kw, c_prev, c_new)
        db: gradient of the cost with respect to the biases b
            (1, 1, 1, c_new)
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
    sh, sw = stride

    # 1. Determine padding dimensions identical to forward prop
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        raise ValueError("padding must be either 'same' or 'valid'")

    # 2. Pad A_prev to align with the forward pass receptive fields
    A_prev_pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # 3. Initialize gradient buffers with zeros
    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)

    # 4. Compute db by summing across batch, height, and width
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # 5. Slide backwards across output spatial dimensions and output channels
    for i in range(h_new):
        v_start = i * sh
        v_end = v_start + kh

        for j in range(w_new):
            h_start = j * sw
            h_end = h_start + kw

            # Extract the receptive field slice across batch and input channels
            # Shape: (m, kh, kw, c_prev)
            slice_A = A_prev_pad[:, v_start:v_end, h_start:h_end, :]

            for c in range(c_new):
                # Isolate scalar gradient per example: shape (m, 1, 1, 1)
                dZ_slice = dZ[:, i, j, c, None, None, None]

                # Accumulate dW: sum over batch m
                dW[:, :, :, c] += np.sum(slice_A * dZ_slice, axis=0)

                # Accumulate dA: distribute gradient back to receptive field
                dA_prev_pad[:, v_start:v_end, h_start:h_end, :] += (
                    W[:, :, :, c] * dZ_slice
                )

    # 6. Unpad dA_prev_pad to recover original input dimensions
    dA_prev = dA_prev_pad[:, ph:ph + h_prev, pw:pw + w_prev, :]

    return dA_prev, dW, db
