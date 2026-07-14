#!/usr/bin/env python3
"""Renames and reformats a DataFrame's Timestamp column."""


def analyze(df):
    """def func"""
    return df.drop(columns=["Timestamp"]).describe()
