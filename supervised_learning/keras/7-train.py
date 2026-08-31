#!/usr/bin/env python3
"""  7. Train  """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                validation_data=None, verbose=True, shuffle=False):
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
    # Create a list of callbacks.
    callbacks = []

    # Early stopping should only happen when:
    # 1. early_stopping is True
    # 2. validation_data exists
    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)

        callbacks.append(early_stopping_callback)

    # Learning rate decay should only happen if validation_data exists.
    if learning_rate_decay and validation_data is not None:

        # Inverse time decay:
        # lr = alpha / (1 + decay_rate * epoch)
        #
        # Keras gives the scheduler a 0-indexed epoch:
        # Epoch 1 -> epoch = 0
        # Epoch 2 -> epoch = 1
        # Epoch 3 -> epoch = 2
        def scheduler(epoch, lr=None):
            return alpha / (1 + decay_rate * epoch)

        # verbose=1 makes Keras print the learning rate update message.
        learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        )

        callbacks.append(learning_rate_callback)
    # Train the model.
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose_mode,
        shuffle=shuffle
    )

    return history
