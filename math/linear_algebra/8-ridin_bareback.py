#!/usr/bin/env python3

"""Matrix multiplication with Python only"""


def mat_mul(mat1, mat2):
    """defining the function"""
    if len(mat1[0]) != len(mat2):
        return None
    else:
        fina = []
        for i in range(len(mat1)):
            iterm = []
            for j in range(len(mat2[0])):
                iterm.append(0)
            fina.append(iterm)
    for i in range(len(mat1)):      # i picks the row   in A (and C)
        for j in range(len(mat2[0])):  # j picks the col   in B (and C)
            for k in range(len(mat1[0])):  # k runs the dot-product sum
                fina[i][j] += mat1[i][k] * mat2[k][j]

    return fina
