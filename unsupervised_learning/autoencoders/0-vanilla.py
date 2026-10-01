#!/usr/bin/env python3
"""Vanilla Autoencoder"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def autoencoder(input_dims, hidden_layers, latent_dims):
    """def the func"""
    # --- Encoder: input_dims -> hidden_layers -> latent_dims ---
    encoder_input = Input(shape=(input_dims,))
    x = encoder_input
    for units in hidden_layers:
        x = Dense(units, activation='relu')(x)
    latent = Dense(latent_dims, activation='relu')(x)
    encoder = Model(encoder_input, latent)

    # --- Decoder: latent_dims -> reversed(hidden_layers) -> input_dims ---
    decoder_input = Input(shape=(latent_dims,))
    x = decoder_input
    for units in reversed(hidden_layers):
        x = Dense(units, activation='relu')(x)
    outputs = Dense(input_dims, activation='sigmoid')(x)
    decoder = Model(decoder_input, outputs)

    # --- Full autoencoder: encoder + decoder chained ---
    auto = Model(encoder_input, decoder(encoder(encoder_input)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
