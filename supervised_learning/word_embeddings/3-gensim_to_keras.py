#!/usr/bin/env python3
"""Converts a Gensim Word2Vec model into a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a trained Gensim Word2Vec model to a Keras Embedding layer."""
    vectors = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=vectors.shape[0],
        output_dim=vectors.shape[1],
        embeddings_initializer=tf.keras.initializers.Constant(vectors),
        trainable=True
    )

    return embedding
