#!/usr/bin/env python3
"""TF-IDF"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """def the func"""
    words_per_sentence = [
        re.findall(r'\b[a-z]+\b', sentence.lower())
        for sentence in sentences
    ]

    if vocab is None:
        features = sorted(set(
            word
            for words in words_per_sentence
            for word in words
        ))
    else:
        features = list(vocab)

    s = len(sentences)
    f = len(features)

    tf = np.zeros((s, f), dtype=float)

    word_to_index = {
        word: index for index, word in enumerate(features)
    }

    for i, words in enumerate(words_per_sentence):
        for word in words:
            if word in word_to_index:
                tf[i, word_to_index[word]] += 1

    # Number of documents containing each word
    df = np.count_nonzero(tf > 0, axis=0)

    # Smoothed inverse document frequency
    idf = np.log((1 + s) / (1 + df)) + 1

    # TF-IDF weights
    embeddings = tf * idf

    # L2 normalization of each sentence vector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Avoid division by zero for empty vectors
    embeddings = np.divide(
        embeddings,
        norms,
        out=np.zeros_like(embeddings),
        where=norms != 0
    )

    return embeddings, np.array(features)
