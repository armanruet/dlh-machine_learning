#!/usr/bin/env python3
"""momentum update"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """One momentum update for a single variable.

    Returns:
        var: updated variable
        v:   updated moment — MUST be returned and passed back
             on the next call, or all memory is lost.
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
