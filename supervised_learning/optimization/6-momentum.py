#!/usr/bin/env python3
"""Momentum Upgrade"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Returns a TensorFlow momentum optimizer."""
    # In Keras, momentum lives inside SGD:
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
