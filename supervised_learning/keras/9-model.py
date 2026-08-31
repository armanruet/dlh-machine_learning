#!/usr/bin/env python3
"""Saves and loads a Keras model."""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model.

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model.
    Returns:
        the loaded model
    """
    return K.models.load_model(filename)
