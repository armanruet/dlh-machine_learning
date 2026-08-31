#!/usr/bin/env python3
"""  4. Train  """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                validation_data=None, epochs, verbose=True, shuffle=False):
    """
    Trains a Keras model using mini-batch gradient descent.

    Arguments:
        network: compiled Keras model
        data: numpy array of shape (m, nx)
        labels: one-hot numpy array of shape (m, classes)
        batch_size: number of samples per batch
        epochs: number of passes through the dataset
        verbose: boolean, whether to print training output
        shuffle: boolean, whether to shuffle batches each epoch

    Returns:
        The History object generated after training.
    """

    # Convert boolean verbose to Keras integer verbosity.
    # True  -> 1, show progress bar
    # False -> 0, no output
    verbose_mode = 1 if verbose else 0

    # Train the model.
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose_mode,
        shuffle=shuffle
    )

    return history
