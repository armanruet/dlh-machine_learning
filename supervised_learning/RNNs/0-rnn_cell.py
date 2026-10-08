#!/usr/bin/env python3
"""Simple RNN"""

import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        """
        Initialize the RNN cell.

        i: dimensionality of the input data
        h: dimensionality of the hidden state
        o: dimensionality of the output
        """
        # Weights: initialized using a standard normal distribution
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Biases: initialized to zeros
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        h_prev: previous hidden state, shape (m, h)
        x_t: current input, shape (m, i)

        Returns:
            h_next: next hidden state, shape (m, h)
            y: output probabilities, shape (m, o)
        """
        # Combine the previous hidden state and current input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Calculate the next hidden state
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Calculate output scores
        z = np.matmul(h_next, self.Wy) + self.by

        # Apply softmax to get output probabilities
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return h_next, y
