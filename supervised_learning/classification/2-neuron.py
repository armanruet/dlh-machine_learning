#!/usr/bin/env python3
"""Neuron that performs binary classification"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize the neuron's attributes"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0.0, 1.0, (1, nx))

        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        ""def FP""
        Z = self.__W @ X + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
