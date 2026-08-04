import sqlite3
import yfinance as yf 
from .filtering_stock import * 
import pandas as pd
import torch 
from torch.utils.data import Dataset
import numpy as np

def df_to_sql(df, returns, type_return):
    """
    Formats the downloaded OHLCV data and matching returns for SQL storage.

    Args:
        df (pd.DataFrame): The DataFrame to format.
        returns (pd.DataFrame): The returns DataFrame.
        type_return (str): The type of return to include in the column name.

    Returns:
        pd.DataFrame: The formatted DataFrame.
    """
    feature_frame = df.stack(level=1).rename_axis(index=['Date', 'Ticker']).reset_index()
    return_frame = returns.stack().rename(type_return).reset_index()
    return_frame.columns = ['Date', 'Ticker', type_return]

    df = feature_frame.merge(return_frame, on=['Date', 'Ticker'], how='inner')
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    df.to_sql(f'{type_return}', conn, if_exists='replace', index=False)

    return df

def load_data(conn, table_name):
    """
    Load and process the data from the database connection.

    Args:
        conn: Database connection object.
        table_name (str): The name of the table to load data from.
    """
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, parse_dates=['Date'])
    df_train = df[df['Date'] < '2016-01-01']
    df_val = df[(df['Date'] >= '2016-01-01') & (df['Date'] < '2019-01-01')]
    df_test = df[df['Date'] >= '2019-01-01'] 

    return df_train, df_val, df_test

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, window_size=30, type_return='overnight_returns'):
        dataframe = dataframe.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

        self.window_size = window_size
        self.target_column = type_return
        self.feature_columns = [
            column for column in dataframe.columns
            if column not in {'Date', 'Ticker', type_return}
        ]
        self.input_size = len(self.feature_columns)

        self.data = []
        self.targets = []
        self.valid_indices = []

        current_idx = 0
        for _, ticker_frame in dataframe.groupby('Ticker', sort=False):
            ticker_frame = ticker_frame.sort_values(by='Date')
            clean_frame = ticker_frame.replace([np.inf, -np.inf], np.nan).fillna(0)

            feature_tensor = torch.tensor(
                clean_frame[self.feature_columns].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            target_tensor = torch.tensor(
                clean_frame[type_return].to_numpy(dtype=np.float32),
                dtype=torch.float32,
            ).unsqueeze(-1)

            self.data.append(feature_tensor)
            self.targets.append(target_tensor)

            count = len(ticker_frame)
            max_start_idx = count - self.window_size

            if max_start_idx > 0:
                for i in range(max_start_idx):
                    self.valid_indices.append(current_idx + i)

            current_idx += count

        if self.data:
            self.data = torch.cat(self.data, dim=0)
            self.targets = torch.cat(self.targets, dim=0)
        else:
            self.data = torch.empty((0, self.input_size), dtype=torch.float32)
            self.targets = torch.empty((0, 1), dtype=torch.float32)
        
    def __len__(self):
        # If we have 100 days and window is 30, we can make 70 windows
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        # Extract the 30-day window
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size
        x_window = self.data[start_idx : end_idx]


        # Extract the target label (the 31st day)
        y_label = self.targets[end_idx]
        
        return x_window, y_label


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

    common_tickers = full_df['Open'].columns.intersection(full_df['Adj Close'].columns)
    full_df = full_df.loc[:, (slice(None), common_tickers)]

    print(f"Data cleaned. Now with {full_df['Adj Close'].shape[1]} tickers after dropping columns with NaN values.")

    open_prices = full_df['Open']
    adj_close_prices = full_df['Adj Close']

    intraday_full_returns = (adj_close_prices - open_prices) / open_prices

    

    intraday_full_returns = df_to_sql(full_df, returns=intraday_full_returns, type_return='intraday_returns')

    overnight_full_returns = ((open_prices - adj_close_prices.shift(1)) / adj_close_prices.shift(1)).dropna()
    
    overnight_full_returns = df_to_sql(full_df, returns=overnight_full_returns, type_return='overnight_returns')


    daily_full_returns = ((adj_close_prices - adj_close_prices.shift(1)) / adj_close_prices.shift(1)).dropna()
    daily_full_returns = df_to_sql(full_df, returns=daily_full_returns, type_return='daily_returns')


