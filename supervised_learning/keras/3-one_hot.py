#!/usr/bin/env python3
"""  0-sequential module  """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """defining the func"""
    return K.utils.to_categorical(labels, num_classes=classes)
