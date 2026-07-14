#!/usr/bin/env python3
"""Renames and reformats a DataFrame's Timestamp column."""


def index(df):
    """def func"""
    df = df.set_index("Timestamp")
    return df
