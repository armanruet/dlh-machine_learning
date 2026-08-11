#!/usr/bin/env python3
"""  0-sequential module  """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a sequential neural network with L2 regularization and Dropout.

    Returns:
        keras.Model: The constructed Keras model.
    """

    model = K.Sequential()
    for i in range(len(layers)):

        dense = K.layers.Dense(
            layers[i],
            input_dim=nx,
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )
        model.add(dense)

        if (i != len(layers) - 1) and (keep_prob is not None):
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
