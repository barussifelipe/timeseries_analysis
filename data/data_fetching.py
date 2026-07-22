import sqlite3
import yfinance as yf 
from .filtering_stock import * 
import torch 
from torch.utils.data import Dataset

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
    df = df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    df.to_sql(f'{type_return}', conn, if_exists='replace', index=True)

    return df

def load_data(conn, table_name):
    """
    Load and process the data from the database connection.

    Args:
        conn: Database connection object.
        table_name (str): The name of the table to load data from.
    """
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, index_col='Date', parse_dates=['Date'])
    df_train = df[df.index < '2016-01-01']
    df_val = df[(df.index >= '2016-01-01') & (df.index < '2019-01-01')]
    df_test = df[df.index >= '2019-01-01'] 

    return df_train, df_val, df_test

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, window_size=30, type_return='intraday_returns'):
        # Convert pandas dataframe to PyTorch tensors
    
        self.data = torch.tensor(dataframe[type_return].values, dtype=torch.float32)
        if self.data.dim() == 1:
            self.data = self.data.unsqueeze(-1)  # Add a feature dimension

        self.window_size = window_size

        self.valid_indices = []
        
        # Count how many rows exist for each ticker
        ticker_counts = dataframe['Ticker'].value_counts(sort=False)
        
        current_idx = 0
        for ticker, count in ticker_counts.items():
            # If a company has 1000 days of data, and window is 30,
            # valid starting indices are 0 through 969.
            max_start_idx = count - self.window_size
            
            if max_start_idx > 0:
                # Add all safe starting indices for this specific ticker
                for i in range(max_start_idx):
                    self.valid_indices.append(current_idx + i)
            
            # Jump the current index forward to the next ticker's starting row
            current_idx += count
        
    def __len__(self):
        # If we have 100 days and window is 30, we can make 70 windows
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        # Extract the 30-day window
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size
        x_window = self.data[start_idx : end_idx]


        # Extract the target label (the 31st day)
        # Assuming the target you want to predict is the first column (index 0)
        y_label = self.data[end_idx, 0].unsqueeze(-1) 
        
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

    print(f"Data cleaned. Now with {full_df['Close'].shape[1]} tickers after dropping columns with NaN values.")

    intraday_full_returns = (full_df['Close'] - full_df['Open'])/full_df['Open']
    intraday_full_returns = df_to_sql(intraday_full_returns, 'intraday_returns')

    overnight_full_returns = ((full_df['Open'] - full_df['Close'].shift(1))/full_df['Close'].shift(1)).dropna()
    overnight_full_returns = df_to_sql(overnight_full_returns, 'overnight_returns')


    daily_full_returns = ((full_df['Close'] - full_df['Close'].shift(1))/full_df['Close'].shift(1)).dropna()
    daily_full_returns = df_to_sql(daily_full_returns, 'daily_returns')


