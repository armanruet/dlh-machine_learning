#!/usr/bin/env python3
"""Saves and loads a model's configuration in JSON format."""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.
        None
    """
    # don't forget good practices for file operations using with
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """
    Loads a model with a specific configuration.
    Returns:
        the loaded model
    """
    # don't forget good practices for file operations using with
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
