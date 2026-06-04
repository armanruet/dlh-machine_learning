#!/usr/bin/env python3
"""Derive happiness in oneself from a good day's work"""


def poly_integral(poly, C=0):
    """defining the function"""
    if not isinstance(C, int) or isinstance(C, bool):
        return None
    poly_f = [C]
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(type(i) in (int, float) for i in poly):
        return None
    for i in range(len(poly)):
        check = poly[i]/(i+1)
        poly_f.append(int(check) if check.is_integer() else check)
    while len(poly_f) > 1 and poly_f[-1] == 0:
        poly_f.pop()
    return poly_f
