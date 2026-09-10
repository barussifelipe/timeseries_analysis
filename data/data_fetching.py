import io
import logging
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone

import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from .filtering_stock import *
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


HISTORY_PERIODS = ('20y', '25y', '30y', 'max')
HISTORY_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
DATA_DIRECTORY = r'D:\DBs\timeseries_analysis'
HISTORY_DB = rf'{DATA_DIRECTORY}\history_coverage.db'
STOCK_DB = rf'{DATA_DIRECTORY}\stock_data.db'


def _create_history_tables(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS raw_history (
            Ticker TEXT NOT NULL,
            Date TEXT NOT NULL,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume REAL,
            PRIMARY KEY (Ticker, Date)
        ) WITHOUT ROWID
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS history_downloads (
            ticker TEXT PRIMARY KEY,
            active INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            first_date TEXT,
            last_date TEXT,
            last_error TEXT,
            updated_at TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS history_coverage (
            ticker TEXT NOT NULL,
            period TEXT NOT NULL,
            before_na INTEGER NOT NULL,
            after_na INTEGER NOT NULL,
            rows_before_na INTEGER NOT NULL,
            rows_after_na INTEGER NOT NULL,
            PRIMARY KEY (ticker, period),
            FOREIGN KEY (ticker) REFERENCES history_downloads(ticker)
        )
    ''')


def _period_coverage(history, period, as_of):
    if period not in HISTORY_PERIODS:
        raise ValueError(f"period must be one of {HISTORY_PERIODS}")

    frame = history.loc[history.index < as_of, HISTORY_COLUMNS]
    cutoff = None if period == 'max' else as_of - pd.DateOffset(years=int(period[:-1]))
    if cutoff is not None:
        frame = frame.loc[frame.index >= cutoff]

    rows_before = len(frame)
    clean = frame.dropna(subset=HISTORY_COLUMNS)
    before_na = rows_before > 0
    after_na = before_na and len(clean) == rows_before

    if cutoff is not None and after_na:
        after_na = (
            clean.index.min() <= cutoff + pd.Timedelta(days=7)
            and clean.index.max() >= as_of - pd.Timedelta(days=7)
        )

    return before_na, after_na, rows_before, len(clean)


def _normalize_history(history):
    if history.empty:
        return history

    history = history.copy()
    history.index = pd.to_datetime(history.index).tz_localize(None)
    return history.sort_index()


def _download_failure_status(error):
    message = error.lower()
    if 'rate limit' in message or '429' in message or 'too many requests' in message:
        return 'rate_limited'
    if 'no price data' in message or 'no timezone found' in message or 'possibly delisted' in message:
        return 'empty'
    if any(marker in message for marker in (
        'timeout', 'timed out', 'connectionerror', 'connection reset',
        'connection aborted', 'remotedisconnected', 'temporarily unavailable',
        'jsondecodeerror',
    )):
        return 'transient'
    return 'failed'


def _raw_history_rows(ticker, history):
    for date, row in history[HISTORY_COLUMNS].iterrows():
        values = [None if pd.isna(value) else float(value) for value in row]
        yield ticker, date.strftime('%Y-%m-%d'), *values


def scan_history_coverage(
    tickers,
    db_filename=HISTORY_DB,
    as_of='2026-01-01',
    delay_seconds=2.0,
    backoff_seconds=(30, 60, 120, 240),
    transient_backoff_seconds=(5, 15),
):
    """Download each ticker's maximum history once and checkpoint coverage."""
    tickers = list(dict.fromkeys(tickers))
    as_of = pd.Timestamp(as_of)

    with closing(sqlite3.connect(db_filename)) as conn, conn:
        _create_history_tables(conn)
        conn.execute('UPDATE history_downloads SET active = 0')
        conn.executemany(
            '''
            INSERT INTO history_downloads (ticker, active)
            VALUES (?, 1)
            ON CONFLICT(ticker) DO UPDATE SET active = 1
            ''',
            ((ticker,) for ticker in tickers),
        )

    next_request_at = time.monotonic()
    for position, ticker in enumerate(tickers, start=1):
        with closing(sqlite3.connect(db_filename)) as conn, conn:
            status, has_raw_data = conn.execute(
                '''
                SELECT d.status, EXISTS (
                    SELECT 1 FROM raw_history r WHERE r.Ticker = d.ticker LIMIT 1
                )
                FROM history_downloads d
                WHERE d.ticker = ?
                ''',
                (ticker,),
            ).fetchone()
        if status == 'success' and has_raw_data:
            continue

        last_error = None
        final_status = 'failed'
        history = pd.DataFrame()

        attempt = 0
        while True:
            attempt += 1
            wait = next_request_at - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            next_request_at = time.monotonic() + delay_seconds

            captured_log = io.StringIO()
            log_handler = logging.StreamHandler(captured_log)
            logging.getLogger('yfinance').addHandler(log_handler)
            try:
                history = _normalize_history(yf.download(
                    ticker,
                    period='max',
                    auto_adjust=True,
                    threads=False,
                    progress=False,
                    multi_level_index=False,
                ))
                missing_columns = set(HISTORY_COLUMNS) - set(history.columns)
                if missing_columns:
                    raise ValueError(f"missing columns: {sorted(missing_columns)}")
                if not history.empty:
                    final_status = 'success'
                    break
                last_error = captured_log.getvalue().strip() or 'Yahoo returned no usable rows'
                final_status = _download_failure_status(last_error)
            except Exception as exc:
                last_error = str(exc)
                final_status = (
                    'rate_limited'
                    if isinstance(exc, YFRateLimitError)
                    else _download_failure_status(last_error)
                )
            finally:
                logging.getLogger('yfinance').removeHandler(log_handler)

            retry_delays = (
                backoff_seconds if final_status == 'rate_limited'
                else transient_backoff_seconds if final_status == 'transient'
                else ()
            )
            if attempt > len(retry_delays):
                if final_status == 'transient':
                    final_status = 'failed'
                break
            time.sleep(retry_delays[attempt - 1])

        updated_at = datetime.now(timezone.utc).isoformat()
        with closing(sqlite3.connect(db_filename)) as conn, conn:
            if final_status == 'success':
                conn.execute('DELETE FROM raw_history WHERE Ticker = ?', (ticker,))
                conn.executemany(
                    '''
                    INSERT INTO raw_history
                        (Ticker, Date, Open, High, Low, Close, Volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''',
                    _raw_history_rows(ticker, history),
                )
                conn.execute('DELETE FROM history_coverage WHERE ticker = ?', (ticker,))
                conn.executemany(
                    '''
                    INSERT INTO history_coverage
                        (ticker, period, before_na, after_na, rows_before_na, rows_after_na)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        (ticker, period, int(before_na), int(after_na), rows_before, rows_after)
                        for period in HISTORY_PERIODS
                        for before_na, after_na, rows_before, rows_after
                        in [_period_coverage(history, period, as_of)]
                    ),
                )
                first_date = history.index.min().isoformat()
                last_date = history.index.max().isoformat()
                last_error = None
            else:
                first_date = last_date = None

            conn.execute(
                '''
                UPDATE history_downloads
                SET status = ?, attempts = ?, first_date = ?, last_date = ?,
                    last_error = ?, updated_at = ?
                WHERE ticker = ?
                ''',
                (
                    final_status,
                    attempt,
                    first_date,
                    last_date,
                    last_error,
                    updated_at,
                    ticker,
                ),
            )
        print(f'[{position}/{len(tickers)}] {ticker}: {final_status}')


def history_lengths(period, db_filename=HISTORY_DB):
    """Return and print ticker counts before and after NA completeness checks."""
    if period not in HISTORY_PERIODS:
        raise ValueError(f"period must be one of {HISTORY_PERIODS}")

    with closing(sqlite3.connect(db_filename)) as conn, conn:
        _create_history_tables(conn)
        requested = conn.execute(
            'SELECT COUNT(*) FROM history_downloads WHERE active = 1'
        ).fetchone()[0]
        before_na, after_na = conn.execute(
            '''
            SELECT COALESCE(SUM(c.before_na), 0), COALESCE(SUM(c.after_na), 0)
            FROM history_coverage c
            JOIN history_downloads d ON d.ticker = c.ticker
            WHERE d.active = 1 AND d.status = 'success' AND c.period = ?
            ''',
            (period,),
        ).fetchone()
        statuses = dict(conn.execute(
            '''
            SELECT status, COUNT(*)
            FROM history_downloads
            WHERE active = 1
            GROUP BY status
            '''
        ))

    result = {
        'period': period,
        'requested': requested,
        'before_na': before_na,
        'after_na': after_na,
        'rate_limited': statuses.get('rate_limited', 0),
        'empty': statuses.get('empty', 0),
        'failed': statuses.get('failed', 0),
        'pending': statuses.get('pending', 0),
    }
    print(result)
    return result

def df_to_sql(extra_features):
    """
    Assembles the per-ticker, per-date feature frames into a single table and
    writes it to SQL. No raw OHLCV data is loaded here - only the already
    computed relative/return features are needed.

    Args:
        extra_features (dict[str, pd.DataFrame]): Per-ticker, per-date feature
            frames (e.g. relative range, log volume, returns) to merge together.
            Only rows present in every feature (inner join) are kept.

    Returns:
        pd.DataFrame: The formatted DataFrame.
    """
    feature_names = list(extra_features)
    first_name = feature_names[0]
    feature_canvas = extra_features[first_name].stack().rename(first_name).reset_index()
    feature_canvas.columns = ['Date', 'Ticker', first_name]

    for feature_name in feature_names[1:]:
        feature_frame = extra_features[feature_name].stack().rename(feature_name).reset_index()
        feature_frame.columns = ['Date', 'Ticker', feature_name]
        feature_canvas = feature_canvas.merge(feature_frame, on=['Date', 'Ticker'], how='inner')

    feature_canvas = feature_canvas.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    feature_canvas.to_sql('features', conn, if_exists='replace', index=False)

    return feature_canvas

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

def fit_zscore_stats(df_train, column='log_volume'):
    """
    Computes per-ticker mean/std for a column using the training split only.

    Fitting on train alone keeps validation/test statistics out of the transform,
    and doing it per-ticker means the z-score answers "is this value unusual for
    THIS stock" rather than encoding how large the stock is.

    Args:
        df_train (pd.DataFrame): The training split.
        column (str): Column to compute statistics for.

    Returns:
        pd.DataFrame: Indexed by Ticker, with 'mean' and 'std' columns.
    """
    return df_train.groupby('Ticker')[column].agg(['mean', 'std'])

def apply_zscore(split_df, stats, column='log_volume'):
    """
    Applies per-ticker z-scoring to a split using train-fitted statistics.

    Tickers missing from the stats (absent in train) or with zero/NaN std are
    left at 0.0, since there is no reliable scale to normalize them against.

    Args:
        split_df (pd.DataFrame): Split to transform (train, val or test).
        stats (pd.DataFrame): Output of fit_zscore_stats.
        column (str): Column to transform.

    Returns:
        pd.DataFrame: A copy of split_df with the column z-scored.
    """
    split_df = split_df.copy()

    mean = split_df['Ticker'].map(stats['mean'])
    std = split_df['Ticker'].map(stats['std'])

    usable = std.notna() & (std != 0) & mean.notna()
    split_df[column] = ((split_df[column] - mean) / std).where(usable, 0.0)

    return split_df

def plot_returns_by_split(df_train, df_val, df_test, type_return='overnight_returns', n_tickers=10):
    """
    Plots returns over time (one line per ticker) for the train, val, and test splits.

    Args:
        df_train, df_val, df_test (pd.DataFrame): Splits returned by load_data.
        type_return (str): Column name of the return to plot.
        n_tickers (int): Number of tickers to plot per split (randomly sampled from the
            tickers present in that split).
    """
    splits = {'Train': df_train, 'Val': df_val, 'Test': df_test}

    fig, axes = plt.subplots(len(splits), 1, figsize=(14, 4 * len(splits)), sharex=False)

    for ax, (split_name, split_df) in zip(axes, splits.items()):
        tickers = split_df['Ticker'].unique()
        sampled_tickers = np.random.choice(tickers, size=min(n_tickers, len(tickers)), replace=False)

        for ticker in sampled_tickers:
            ticker_frame = split_df[split_df['Ticker'] == ticker].sort_values('Date')
            ax.plot(ticker_frame['Date'], ticker_frame[type_return], label=ticker, linewidth=0.8)

        ax.set_title(f'{split_name} — {type_return}')
        ax.set_xlabel('Date')
        ax.set_ylabel(type_return)
        ax.legend(loc='upper right', fontsize='small', ncol=2)

    fig.tight_layout()
    plt.show()

    return fig


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, window_size=30, type_return='overnight_returns'):
        dataframe = dataframe.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

        self.window_size = window_size
        self.target_column = type_return
        self.feature_columns = [
            column for column in dataframe.columns
            if column not in {'Date', 'Ticker'}
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
        x_window = self.data[start_idx : end_idx] #To test only with the returns. 


        # Extract the target label (the 31st day)
        y_label = self.targets[end_idx]
        
        return x_window, y_label


if __name__ == "__main__":

    db_filename = STOCK_DB

    conn = sqlite3.connect(db_filename)

    cursor = conn.cursor()

    path_nasdaq = rf"{DATA_DIRECTORY}\stock_overview_NASDAQ.csv"
    path_nyse = rf"{DATA_DIRECTORY}\stock_overview_NYSE.csv"

    index_nyse = fetch_index_stock(path_nyse, "NYSE")
    index_nasdaq = fetch_index_stock(path_nasdaq, "NASDAQ")

    index = index_nyse + index_nasdaq

    full_df = yf.download(index, start="2006-01-01", end="2026-01-01", group_by='column', auto_adjust=True, threads=True, progress=True)

    print(f"Downloaded data for {len(index)} tickers. Saving to database...")

    full_df = full_df.dropna(how='any', axis=1)

    common_tickers = full_df['Open'].columns.intersection(full_df['Close'].columns)
    full_df = full_df.loc[:, (slice(None), common_tickers)]

    print(f"Data cleaned. Now with {full_df['Close'].shape[1]} tickers after dropping columns with NaN values.")

    open_prices = full_df['Open']
    high_prices = full_df['High']
    low_prices = full_df['Low']
    close_prices = full_df['Close']
    volume = full_df['Volume']

    intraday_full_returns = (close_prices - open_prices) / open_prices
    overnight_full_returns = ((open_prices - close_prices.shift(1)) / close_prices.shift(1)).dropna()
    
    # Relative OHLCV features: kept scale-free/stationary so they sit on the
    # same footing as the returns instead of raw, non-stationary price/volume levels.
    range_pct = (high_prices - low_prices) / close_prices
    log_volume = np.log(volume.replace(0, np.nan))

    extra_features = {
                'range_pct': range_pct,
                'log_volume': log_volume,
                'overnight_returns': overnight_full_returns,
                'intraday_returns': intraday_full_returns,
                           }

    feature_df = df_to_sql(extra_features=extra_features)
