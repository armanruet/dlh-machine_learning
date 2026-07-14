#!/usr/bin/env python3
import pandas as pd
"""def"""


def rename(df):
    """defining the func"""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    print(df[["Datetime", "Close"]])
    return df
