#!/usr/bin/env python3

"""Function to determine the Determinant"""
def determinant(matrix):
    """defining the function"""
    if matrix == [[]]:
        return 1
    if not matrix:
        return TypeError ("matrix must be a list of lists")
    if len(matrix)!= len(matrix[0]):
        return ValueError ("matrix must be a square matrix")
    if len(matrix)==2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
mat0 = [[]]    
mat5 = []
mat6 = [[1, 2, 3], [4, 5, 6]]
print(determinant(mat0))
print(determinant(mat6))
print(determinant(mat5))
