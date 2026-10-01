#!/usr/bin/env python3
"""COnvolutional Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    def the func
    """
    # --- Encoder: input_dims -> hidden_layers -> latent_dims ---
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for units in filters:
        # Conv2D: 3x3 kernel, same padding, ReLU
        x = keras.layers.Conv2D(filters=units, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        # MaxPooling2D: 2x2 window, stride 2 -> halves spatial dims
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    flat = keras.layers.Flatten()(x)
    out = keras.layers.Dense(latent_dims)(flat)
    return keras.Model(encoder_input, out)
