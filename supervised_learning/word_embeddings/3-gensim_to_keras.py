#!/usr/bin/env python3
"""Converts a Gensim Word2Vec model to a Keras Embedding layer."""

import keras


def gensim_to_keras(model):
    """Convert a trained Word2Vec model to a trainable Keras embedding."""

    vectors = model.wv.vectors

    return keras.layers.Embedding(
        input_dim=vectors.shape[0],
        output_dim=vectors.shape[1],
        embeddings_initializer=keras.initializers.Constant(vectors),
        trainable=True
    )
