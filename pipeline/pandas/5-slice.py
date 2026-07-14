#!/usr/bin/env python3
"""Renames and reformats a DataFrame's Timestamp column."""


def slice(df):
    """def func"""
    return df[["High", "Low", "Close", "Volume_(BTC)"]][::60]
