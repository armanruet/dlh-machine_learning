#!/usr/bin/env python3
"""Moving Average"""


def moving_average(data, beta):
    """def the func"""
    moving_averages = []
    v = 0.0

    for t, value in enumerate(data, start=1):
        v = beta * v + (1 - beta) * value
        corrected = v / (1 - beta ** t)
        moving_averages.append(corrected)

    return moving_averages
