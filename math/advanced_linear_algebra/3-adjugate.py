#!/usr/bin/env python3
"""Function to calculate the Adjugate matrix"""


def determinant(matrix):
    """Returns the determinant of a matrix"""
    if matrix == [[]]:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    total = 0
    for j in range(len(matrix)):
        sub = [[matrix[i][k] for k in range(len(matrix)) if k != j]
               for i in range(1, len(matrix))]
        total += ((-1) ** j) * matrix[0][j] * determinant(sub)
    return total


def adjugate(matrix):
    """Returns the adjugate matrix of a given matrix"""
    if (not isinstance(matrix, list) or len(matrix) == 0
            or not all(isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    cofactor = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            sub = [[matrix[r][c] for c in range(len(matrix)) if c != j]
                   for r in range(len(matrix)) if r != i]
            row.append(((-1) ** (i + j)) * determinant(sub))
        cofactor.append(row)
    return [[cofactor[j][i] for j in range(len(matrix))]
            for i in range(len(matrix))]
