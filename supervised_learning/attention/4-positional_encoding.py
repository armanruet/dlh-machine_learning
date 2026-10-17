#!/usr/bin/env python3
"""
Module positional_encoding
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    def the func
    """
    PE = np.zeros((max_seq_len, dm))

    # Tableau des positions : forme (max_seq_len, 1)
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Terme d'échelle pour la fréquence : forme (1, dm)
    div_term = np.exp(
        np.arange(0, dm, 2) * -(np.log(10000.0) / dm)
    )

    # Application du sinus aux indices pairs (2i)
    PE[:, 0::2] = np.sin(position * div_term)

    # Application du cosinus aux indices impairs (2i + 1)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE
