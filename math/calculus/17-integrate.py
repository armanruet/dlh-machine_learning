#!/usr/bin/env python3
"""Derive happiness in oneself from a good day's work"""


def poly_integral(poly, C=0):
    """defining the function"""
    poly_f = [C]
    if not isinstance(poly, list) and not isinstance(C, int)\
            or not all(isinstance(i, int) for i in poly):
        return None
    if len(poly) == 1:
        return poly
    for i in range(0, len(poly)):
        check = poly[i]/(i+1)
        poly_f.append(int(check) if check.is_integer() else check)
    if all(x == 0 for x in poly_f):
        return [0]
    return poly_f
