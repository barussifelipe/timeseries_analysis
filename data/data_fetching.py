import sqlite3
import numpy as np 
import pandas as pd 
import yfinance as yf 
import matplotlib.pyplot as plt
import os 
from filtering_stock import * 


def df_to_sql(df, type_return):
    """
    Formats the DataFrame to be compatible with SQL storage.

    Args:
        df (pd.DataFrame): The DataFrame to format.
        type_return (str): The type of return to include in the column name.

    Returns:
        pd.DataFrame: The formatted DataFrame.
    """

    df = df.stack().reset_index()
    df.columns = ['Date', 'Ticker', f'{type_return}']
    df.to_sql(f'{type_return}', conn, if_exists='replace', index=True)

    return df

if __name__ == "__main__":

    db_filename = "data/src/stock_data.db"

    conn = sqlite3.connect(db_filename)

    cursor = conn.cursor()

    path_nasdaq = "data/src/stock_overview_NASDAQ.csv"
    path_nyse = "data/src/stock_overview_NYSE.csv"

    index_nyse = fetch_index_stock(path_nyse, "NYSE")
    index_nasdaq = fetch_index_stock(path_nasdaq, "NASDAQ")

    index = index_nyse + index_nasdaq

    full_df = yf.download(index, start="2006-01-01", end="2026-01-01", group_by='column', auto_adjust=False, threads=True, progress=True)

    print(f"Downloaded data for {len(index)} tickers. Saving to database...")

    full_df = full_df.dropna(how='any', axis=1)

    print(f"Data cleaned. Now with {full_df['Close'].shape[1]} tickers after dropping columns with NaN values.")

    intraday_full_returns = (full_df['Close'] - full_df['Open'])/full_df['Open']
    intraday_full_returns = df_to_sql(intraday_full_returns, 'intraday_returns')

    overnight_full_returns = ((full_df['Open'] - full_df['Close'].shift(1))/full_df['Close'].shift(1)).dropna()
    overnight_full_returns = df_to_sql(overnight_full_returns, 'overnight_returns')


    daily_full_returns = ((full_df['Close'] - full_df['Close'].shift(1))/full_df['Close'].shift(1)).dropna()
    daily_full_returns = df_to_sql(daily_full_returns, 'daily_returns')


