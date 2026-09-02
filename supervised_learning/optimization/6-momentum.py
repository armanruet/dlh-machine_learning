#!/usr/bin/env python3
"""Momentum Upgrade"""
import tensorflow.keras as K


def create_momentum_op(alpha, beta1):
    """Returns a TensorFlow momentum optimizer."""
    # In Keras, momentum lives inside SGD:
    optimizer = K.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1)
    return optimizer
