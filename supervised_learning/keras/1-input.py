#!/usr/bin/env python3
"""  0-sequential module  """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds a sequential neural network with L2 regularization and Dropout.

    Returns:
        keras.Model: The constructed Keras model.
    """

    inputs = K.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    return K.Model(inputs=inputs, outputs=x)
