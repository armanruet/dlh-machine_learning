#!/usr/bin/env python3
"""Renames and reformats a DataFrame's Timestamp column."""


def fill(df):
    """def func"""
    df = df.drop(columns=['Weighted_Price'])
    df["Close"] = df["Close"].ffill()
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])
    df["Open"] = df["Open"].fillna(df["Close"])
    df[["Volume_(BTC)", "Volume_(Currency)"]] = df[[
        "Volume_(BTC)", "Volume_(Currency)"]].fillna(0)
    return df
