#!/usr/bin/env python3
"""file to DF"""
import pandas as pd


def from_file(filename, delimiter):
    """def the func"""
    return pd.read_csv(filename, sep=delimiter)
