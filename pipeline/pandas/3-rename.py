#!/usr/bin/env python3
"""Module that renames and reformats a DataFrame's Timestamp column."""
import pandas as pd


def rename(df):
    """Rename Timestamp to Datetime, convert to datetime, keep Datetime/Close."""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    df = df[["Datetime", "Close"]]
    return df
