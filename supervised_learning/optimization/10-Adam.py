#!/usr/bin/env python3
"""
Adam Upgraded
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """function that implements Adam gradient descent in tensorflow"""
    return tf.keras.optimizers.Adam(
        learning_rate=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
