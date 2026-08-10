#!/usr/bin/env python3
"""0-sequential.py — Build a neural network with Keras Sequential API."""
import tensorflow as tf
from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Build a Sequential neural network.

    Args:
        nx:          number of input features
        layers:      list of node counts per layer  e.g. [256, 256, 10]
        lambtha:     L2 regularization parameter
        keep_prob:   probability a node is KEPT by dropout (1 - drop_rate)

    Returns:
        keras.Model
    """
    # Build a Sequential model — one lane, layers added in order
    network = keras.Sequential()

    # ─── Add each Dense + Dropout pair ───
    for i in range(len(layers)):
        # First layer MUST be told input_shape (we can't use Input class)
        if i == 0:
            network.add(keras.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=keras.regularizers.l2(lambtha),
                input_shape=(nx,)
            ))
        else:
            network.add(keras.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=keras.regularizers.l2(lambtha)
            ))

        # Dropout AFTER every Dense EXCEPT the last (output) layer.
        # Why? The output layer's activations ARE the predictions —
        # dropping them would corrupt the class probabilities.
        if i != len(layers) - 1:
            # `rate` = fraction to DROP. We have keep_prob (fraction to KEEP),
            # so drop_rate = 1 - keep_prob.
            network.add(keras.layers.Dropout(1 - keep_prob))

    return network
