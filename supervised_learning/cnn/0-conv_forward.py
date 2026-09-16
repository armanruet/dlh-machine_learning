#!/usr/bin/env python3
"""Performs forward propagation over a convolutional layer."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """defining the func"""
    # 1. Unpack dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # 2. Decide padding
    if padding == "valid":
        ph, pw = 0, 0
    else:  # "same"
        ph = max(0, ((h_prev - 1) * sh + kh - h_prev) // 2)
        pw = max(0, ((w_prev - 1) * sw + kw - w_prev) // 2)

    # 3. Zero-pad (never pad batch or channel axes)
    A_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode="constant")

    # 4. Output size
    out_h = (h_prev + 2 * ph - kh) // sh + 1
    out_w = (w_prev + 2 * pw - kw) // sw + 1

    # 5. Slide the window
    Z = np.zeros((m, out_h, out_w, c_new))
    for i in range(out_h):                  # output row
        for j in range(out_w):              # output column
            r = i * sh                      # window's top row in padded input
            c = j * sw                      # window's left col
            patch = A_pad[:, r:r + kh, c:c + kw, :]      # (m, kh, kw, c_prev)
            for k in range(c_new):                       # each kernel
                Z[:, i, j, k] = np.sum(patch * W[:, :, :, k],
                                       axis=(1, 2, 3))   # (m,) — one per image

    # 6. Bias (broadcasts) then activation
    return activation(Z + b)
