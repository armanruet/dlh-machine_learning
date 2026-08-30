#!/usr/bin/env python3
"""  0-sequential module  """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Builds a sequential neural network with L2 regularization and Dropout.

    Returns:
        keras.Model: The constructed Keras model.
    """
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2)
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return None
