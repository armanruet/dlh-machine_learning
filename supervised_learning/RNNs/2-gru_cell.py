#!/usr/bin/env python3
"""GRU"""
import numpy as np


class GRUCell:
    """GRUCELL"""
    def __init__(self, i, h, o):
        """Init"""
        # Weights initialized in the required order
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        # Biases initialized to zeros
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """FOrward Pass"""
        # Combine previous hidden state and current input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = 1 / (1 + np.exp(-(
            np.matmul(concat, self.Wz) + self.bz
        )))

        # Reset gate
        r = 1 / (1 + np.exp(-(
            np.matmul(concat, self.Wr) + self.br
        )))

        # Candidate hidden state
        reset_hidden = r * h_prev
        candidate_input = np.concatenate(
            (reset_hidden, x_t), axis=1
        )

        h_tilde = np.tanh(
            np.matmul(candidate_input, self.Wh) + self.bh
        )

        # Combine previous state and candidate state
        h_next = z * h_prev + (1 - z) * h_tilde

        # Calculate output scores
        q = np.matmul(h_next, self.Wy) + self.by

        # Apply numerically stable softmax
        exp_q = np.exp(q - np.max(q, axis=1, keepdims=True))
        y = exp_q / np.sum(exp_q, axis=1, keepdims=True)

        return h_next, y
