#!/usr/bin/env python3
"""Fuction returns array addition"""


def add_matrices2D(mat1, mat2):
    """defining the dunction"""
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        x = []
        for i in range(len(mat1)):
            j = []
            for k in range(len(mat1[0])):
                j.append(mat1[i][k]+mat2[i][k])
            x.append(j)
        return x
    else:
        return None
