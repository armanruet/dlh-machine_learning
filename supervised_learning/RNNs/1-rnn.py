#!/usr/bin/env python3
"""forward propagation for a simple RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN.

    rnn_cell: RNNCell instance
    X: input data of shape (t, m, i)
    h_0: initial hidden state of shape (m, h)

    Returns:
        H: all hidden states, shape (t + 1, m, h)
        Y: all outputs, shape (t, m, o)
    """
    # Extract the dimensions of the input sequence
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    # Allocate memory for all hidden states, including h_0
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Allocate memory for the output at every time step
    Y = np.zeros((t, m, o))

    # Process the sequence one time step at a time
    for k in range(t):
        H[k + 1], Y[k] = rnn_cell.forward(H[k], X[k])

    return H, Y
