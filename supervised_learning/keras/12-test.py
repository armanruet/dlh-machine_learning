#!/usr/bin/env python3
"""Tests a neural network."""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
        the loss and accuracy of the model with the testing data,
        respectively
    """
    # test the model, it is only trained and validated until now
    return network.evaluate(x=data, y=labels, verbose=verbose)
