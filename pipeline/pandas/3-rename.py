#!/usr/bin/env python3
"""Renames and reformats a DataFrame's Timestamp column."""
import pandas as pd


def rename(df):
    """Rename keep Datetime/Close."""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    df = df[["Datetime", "Close"]]
    return df
