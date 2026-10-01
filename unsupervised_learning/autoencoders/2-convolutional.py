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
    encoder = keras.Model(encoder_input, x)
# ---- Decoder: latent vector -> reconstructed image ----
    decoder_input = keras.Input(shape=(latent_dims,))
    y = decoder_input
    rev = filters[::-1]
    for i, f in enumerate(rev):
        padding = 'valid' if i == len(rev) - 1 else 'same'
        y = keras.layers.Conv2D(f, (3, 3), padding=padding,
                                activation='relu')(y)
        y = keras.layers.UpSampling2D((2, 2))(y)
    outputs = keras.layers.Conv2D(input_dims[-1], (3, 3), padding='same',
                                  activation='sigmoid')(y)
    decoder = keras.Model(decoder_inputs, outputs)

    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
