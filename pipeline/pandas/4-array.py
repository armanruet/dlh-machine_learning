#!/usr/bin/env python3
"""Renames and reformats a DataFrame's Timestamp column."""
import pandas as pd


def array(df):
    """def func"""
    return df[["High", "Close"]].tail(10).to_numpy()
