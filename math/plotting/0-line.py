#!/usr/bin/env python3
"""Creating the first plot"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """defining the function"""

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, 10)
    plt.plot(y, color='red')
