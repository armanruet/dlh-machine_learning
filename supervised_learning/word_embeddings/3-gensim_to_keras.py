#!/usr/bin/env python3
"""Converts a Gensim Word2Vec model into a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a Gensim Word2Vec model to a trainable Keras Embedding."""
    weights = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        weights.shape[0],
        weights.shape[1],
        trainable=True
    )

    embedding.build((None,))
    embedding.set_weights([weights])

    return embedding
