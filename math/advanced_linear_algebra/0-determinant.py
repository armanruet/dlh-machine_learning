#!/usr/bin/env python3

"""Function to determine the Determinant"""


def determinant(matrix):
    """defining the function"""
    if matrix == [[]]:
        return 1
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    total = 0
    for j in range(len(matrix)):
        # Minor: delete row 0, col j
        minor = [[matrix[i][k] for k in range(len(matrix))
                  if k != j]
                 for i in range(1, len(matrix))]
        sign = (-1) ** j
        total += sign * matrix[0][j] * determinant(minor)
    return total
