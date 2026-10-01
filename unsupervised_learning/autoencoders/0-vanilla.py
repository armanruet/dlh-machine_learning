#!/usr/bin/env python3
"""Vanilla Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    def the func
    """
    # --- Encoder: input_dims -> hidden_layers -> latent_dims ---
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(encoder_input, latent)

    # --- Decoder: latent_dims -> reversed(hidden_layers) -> input_dims ---
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, outputs)

    # --- Full autoencoder: encoder + decoder chained ---
    auto = keras.Model(encoder_input, decoder(encoder(encoder_input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
