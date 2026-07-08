#!/usr/bin/env python3

"""function  from_numpy"""
import pandas as pd


def from_numpy(array):
    """defining the function"""
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return pd.DataFrame(array, columns=list(abc[:len(array[0])]))
