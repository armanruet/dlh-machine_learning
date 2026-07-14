#!/usr/bin/env python3
"""def"""
import pandas as pd


def rename(df):
    """defining the func"""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[["Datetime", "Close"]]
    print(df[["Datetime", "Close"]])
    return df
