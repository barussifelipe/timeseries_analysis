"""Raw adjusted-OHLC history for new neural variance runs."""

import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd


def load_raw_neural(database, ticker=None, limit_tickers=None, limit_rows=None):
    if limit_rows is not None:
        raise ValueError('raw-history cohort does not support row truncation')
    cohort = """SELECT Ticker FROM raw_history GROUP BY Ticker
        HAVING SUM(Date < '2016-01-01') > 80
           AND SUM(Date = '2025-12-31') > 0"""
    with closing(sqlite3.connect(database)) as conn:
        names = [row[0] for row in conn.execute(cohort)]
        if ticker is not None:
            names = [name for name in names if name == ticker]
        if limit_tickers is not None:
            names = names[:limit_tickers]
        if not names:
            raise ValueError('no eligible raw-history tickers')
        # Read each ticker in index order; SQLite's host-parameter limit varies by build.
        frames = [raw_frame(pd.read_sql_query("""SELECT Ticker, Date, Open, High, Low, Close, Volume
                    FROM raw_history WHERE Ticker = ? AND Date < '2026-01-01' ORDER BY Date""",
                    conn, params=(name,))) for name in names]
    frame = pd.concat(frames, ignore_index=True)
    frame['Ticker'] = frame.Ticker.astype('category')
    frame['Date'] = pd.to_datetime(frame.Date)
    variance = frame.RawVariance.to_numpy(dtype=float)
    valid = frame.Valid.to_numpy(dtype=bool)
    training = valid & (frame.Date.to_numpy() < np.datetime64('2016-01-01')) & (variance > 0)
    if not training.any():
        raise ValueError('no positive pre-2016 raw variance')
    floor = float(variance[training].min())
    frame['Variance'] = np.where(valid & (variance == 0), floor, variance)
    history = {name: group.Variance.to_numpy(dtype=float) for name, group in
               frame.loc[valid & (frame.Date < '2016-01-01')].groupby('Ticker') if len(group)}
    return frame, history, floor


def raw_frame(frame):
    prices = frame[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy(dtype=float)
    o, h, l, c, volume = prices.T
    valid = (np.isfinite(prices).all(axis=1) & (prices[:, :4] > 0).all(axis=1)
             & (volume >= 0) & (h >= np.maximum(o, c)) & (l <= np.minimum(o, c)))
    variance = np.full(len(frame), np.nan)
    intraday = np.full(len(frame), np.nan)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        variance[valid] = .5 * np.log(h[valid] / l[valid]) ** 2 - (2 * np.log(2) - 1) * np.log(c[valid] / o[valid]) ** 2
        intraday[valid] = np.log(c[valid] / o[valid])
    valid &= np.isfinite(variance) & (variance >= 0) & np.isfinite(intraday)
    return pd.DataFrame({'Ticker': frame.Ticker, 'Date': frame.Date, 'Valid': valid,
                         'RawVariance': variance, 'IntradayLogReturn': intraday})
