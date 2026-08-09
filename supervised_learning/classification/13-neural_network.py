#!/usr/bin/env python3
"""
Write a class that defines a neural network with one hidden layer
"""

import numpy as np


class NeuralNetwork:
    """

All exceptions should be raised in the order listed above
    """

    def __init__(self, nx, nodes):
        """
Upon instantiation, it should be initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # nodes checks
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # init neuron "Hidden Layer" x nodes
        # weights (W1) is of shape (nodes, nx), drawing from std normal
        self.__W1 = np.random.standard_normal(size=(nodes, nx))
        # neutral bias (b) init
        self.__b1 = np.zeros((nodes, 1))
        # neuron answer (A) init
        self.__A1 = 0
        # init neuron 2 "Output Layer"
        # weights (W2) is of shape (1, nodes), drawing from std normal
        self.__W2 = np.random.standard_normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """ the private attributes __A1 and __A2, respectively
        """
        # calculate z1, weights x input plus bias
        z1 = np.matmul(self.__W1, X) + self.__b1
        # squeeze z1 using sigmoid function between (0, 1)
        self.__A1 = 1/(1+np.exp(-z1))

        # only dimensions change, but np broadcasting handles it
        # calculate z2, uses A1 as input!
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        # squeeze z2 using sigmoid function between (0, 1)
        self.__A2 = 1/(1+np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression."""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) +
                       (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """Evaluate the neural network's predictions."""
        A1, A2 = self.forward_prop(X)
        return np.where(A2 >= 0.5, 1, 0), self.cost(Y, A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate one pass of gradient descent on the network."""
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1


