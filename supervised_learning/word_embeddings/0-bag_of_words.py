#!/usr/bin/env python3
"""Bag of words"""
import numpy as np
import string
import re


def bag_of_words(sentences, vocab=None):
    """Def the func"""
    # Step A: Build the vocabulary if not provided
    if vocab is None:
        vocab = []
        for sentence in sentences:
            for word in re.findall(r'\w+', sentence.lower()):
                if len(word) >= 2 and word not in vocab:
                    vocab.append(word)
        # Sort the vocabulary alphabetically
        vocab = sorted(vocab)

    # Step B: Create the embedding matrix
    s = len(sentences)
    f = len(vocab)
    embeddings = np.zeros((s, f))

    # Step C: Fill in counts for each sentence
    for i, sentence in enumerate(sentences):
        counts = {}
        for word in re.findall(r'\w+', sentence.lower()):
            if len(word) >= 2:
                if word in counts:
                    counts[word] = counts[word] + 1
                else:
                    counts[word] = 1

        for word, count in counts.items():
            idx = vocab.index(word)
            embeddings[i][idx] = count

    return embeddings, vocab
