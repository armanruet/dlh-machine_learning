#!/usr/bin/env python3
"""Fuction returns array addition"""


def add_arrays(arr1, arr2):
    """defining the dunction"""
    if len(arr1) == len(arr2):
        x = []
        for i in range(len(arr1)):
            x.append(arr1[i]+arr2[i])
        return x
    else:
        return None
