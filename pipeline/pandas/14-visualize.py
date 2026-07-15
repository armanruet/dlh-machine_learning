#!/usr/bin/env python3
"""Visualization"""
"""importing the modules"""
import matplotlib.pyplot as plt
import pandas as pd
"""importing 2-from_file"""
from_file = __import__('2-from_file').from_file
"""loading the csv"""
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# dropring the column
df.drop(columns=['Weighted_Price'], inplace=True)
# renaming the column
df.rename(columns={'Timestamp': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], unit='s')
# setting the indexes
df.set_index('Date', inplace=True)
df['Close'].fillna(method='ffill', inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

df = df[df.index >= '2017-01-01']
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})
# printing the dataframe
print(df)
# plotting the Close column w.r.t indexes
plt.plot(df.index, df['Close'])
# showing the plot
plt.show()
