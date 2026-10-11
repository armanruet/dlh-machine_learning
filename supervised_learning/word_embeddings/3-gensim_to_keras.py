#!/usr/bin/env python3
"""Converts a Gensim Word2Vec model into a Keras Embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a trained Gensim Word2Vec model to a Keras Embedding layer."""
    return tf.keras.layers.Embedding(
        input_dim=len(model.wv),
        output_dim=model.wv.vector_size,
        embeddings_initializer=tf.keras.initializers.Constant(
            model.wv.vectors
        ),
        trainable=True
    )
