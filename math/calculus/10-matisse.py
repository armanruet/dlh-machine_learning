#!/usr/bin/env python3
"""Derive happiness in oneself from a good day's work"""


def poly_derivative(poly):
    """defining the function"""
    poly_f = []
    if not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    for i in range(1, len(poly)):
        if poly[i] == 0:
            poly_f.append(0)
        else:
            poly_f.append(poly[i]*i)
    if all(x == 0 for x in poly_f):
        return [0]
    return poly_f
