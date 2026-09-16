#!/usr/bin/env python3
"""Module that builds a modified version of the LeNet-5 in Keras."""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using Keras.

    Parameters:
        X: K.Input of shape (m, 28, 28, 1) containing the input images

    Returns:
        A compiled K.Model using Adam optimization and accuracy metrics
    """
    # Shared initializer factory with reproducible seed
    init = K.initializers.HeNormal(seed=0)

    # 1. C1: Conv2D with 6 filters (5x5), same padding, relu
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init
    )(X)

    # 2. S2: Max Pooling (2x2) with stride (2x2)
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # 3. C3: Conv2D with 16 filters (5x5), valid padding, relu
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init
    )(pool1)

    # 4. S4: Max Pooling (2x2) with stride (2x2)
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # 5. Flatten 3D feature maps (5x5x16) into a 1D vector (400)
    flat = K.layers.Flatten()(pool2)

    # 6. F5: Fully connected layer with 120 units, relu
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flat)

    # 7. F6: Fully connected layer with 84 units, relu
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(fc1)

    # 8. Output: Fully connected layer with 10 units, softmax
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(fc2)

    # Construct the computational graph model
    model = K.Model(inputs=X, outputs=output)

    # Compile with Adam optimizer, categorical crossentropy, and accuracy
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
