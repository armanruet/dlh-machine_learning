#!/usr/bin/env python3
"""Numpy.array add, subtract, multiply, devide"""


def np_elementwise(mat1, mat2):
    """defining the function"""
    return (mat1+mat2, mat1-mat2, mat1*mat2, mat1/mat2)


mat1 = np.array([[11, 22, 33], [44, 55, 66]])
mat2 = np.array([[1, 2, 3], [4, 5, 6]])
