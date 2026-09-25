"""Shared, ticker-safe one-day variance training data and scoring."""

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SPLITS = {'all': (None, None), 'train': (None, '2016-01-01'), 'val': ('2016-01-01', '2019-01-01'),
          'test': ('2019-01-01', '2026-01-01'), 'crypto': (None, None)}
ESTIMATORS = {'parkinson': 'parkinson', 'garman-klass': 'garman_klass'}


class Forecast(NamedTuple):
    """One target date; actual is the scoring target, prediction is daily variance."""

    ticker: str
    target_date: str
    actual_variance: float
    predicted_variance: float


class TimeSeriesDataset(Dataset):
    """Next-observation windows, stored as one tensor sorted by ticker and date.

    For target row i, input rows are [i - window_size, i) from the same ticker.
    The split filters target dates only, so validation and test windows can use
    earlier observed history. Variance targets must be finite and positive.
    ``consecutive_sessions`` requires all input-to-target steps to be
    adjacent in the panel's observed market-date calendar.
    ``positive_target=False`` preserves the older returns reference behavior.
    """

    def __init__(self, dataframe, window_size=30, feature_columns=('Variance',),
                 target_column='Variance', split='train', positive_target=True,
                 consecutive_sessions=False, session_calendar=None, valid_column=None):
        if window_size < 1 or split not in SPLITS:
            raise ValueError('invalid window size or split')
        self.window_size = window_size
        self.feature_columns = tuple(feature_columns)
        self.target_column = target_column
        self.input_size = len(self.feature_columns)
        needed = ['Ticker', 'Date', *self.feature_columns, target_column]
        if any(c not in dataframe for c in needed):
            raise ValueError('missing dataset column')
        frame = dataframe.copy()
        if not positive_target:
            frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
        frame['Date'] = pd.to_datetime(frame['Date'])
        frame = frame.sort_values(['Ticker', 'Date'])
        if frame.duplicated(['Ticker', 'Date']).any():
            raise ValueError('duplicate ticker date')
        valid = frame[valid_column].to_numpy(dtype=bool) if valid_column else np.ones(len(frame), dtype=bool)
        numbers = frame[list(set(self.feature_columns) | {target_column})].to_numpy(dtype=float)
        if not np.isfinite(numbers[valid]).all() or (positive_target and (frame.loc[valid, target_column].to_numpy(dtype=float) <= 0).any()):
            raise ValueError('variance data must be finite and targets positive')
        if valid_column:
            frame.loc[~valid, list(set(self.feature_columns) | {target_column})] = 0.
        self.data = torch.tensor(frame[list(self.feature_columns)].to_numpy(dtype=np.float32))
        self.targets = torch.tensor(frame[target_column].to_numpy(dtype=np.float32)).unsqueeze(1)
        if (not torch.isfinite(self.data).all() or not torch.isfinite(self.targets).all()
                or (positive_target and (self.targets[torch.tensor(valid.copy())] <= 0).any())):
            raise ValueError('values cannot be represented as finite positive float32 variance')
        self.ticker_names = list(frame['Ticker'].drop_duplicates())
        indices, ids, dates = [], [], []
        start, end = SPLITS[split]
        calendar = None
        if consecutive_sessions:
            calendar = np.sort(pd.to_datetime(session_calendar).to_numpy()) if session_calendar is not None else np.sort(frame['Date'].unique())
            observed = frame['Date'].unique()
            ranks = np.searchsorted(calendar, observed)
            if (not len(calendar) or np.any(ranks == len(calendar))
                    or np.any(calendar[ranks] != observed)):
                raise ValueError('session calendar must contain every observed date')
        offset = 0
        for ticker_id, (ticker, group) in enumerate(frame.groupby('Ticker', sort=False)):
            positions = np.arange(window_size, len(group))
            all_dates = group['Date'].to_numpy()
            group_dates = all_dates[positions]
            mask = np.ones(len(positions), dtype=bool)
            if valid_column:
                good_input = valid[offset:offset + len(group)] & (group['RawVariance'].to_numpy(dtype=float) > 0)
                bad = np.r_[0, np.cumsum(~good_input)]
                mask &= (bad[positions] == bad[positions - window_size]) & valid[offset + positions]
            if consecutive_sessions:
                ranks = np.searchsorted(calendar, all_dates)
                mask &= ranks[window_size:] - ranks[:-window_size] == window_size
            if start is not None:
                mask &= group_dates >= np.datetime64(start)
            if end is not None:
                mask &= group_dates < np.datetime64(end)
            positions = positions[mask]
            indices.append(offset + positions - window_size)
            ids.append(np.full(len(positions), ticker_id, dtype=np.int32))
            dates.append(group_dates[mask])
            offset += len(group)
        self.valid_indices = torch.from_numpy(np.concatenate(indices).astype(np.int64)) if indices else torch.empty(0, dtype=torch.int64)
        self.target_indices = self.valid_indices + window_size
        self.ticker_ids = torch.from_numpy(np.concatenate(ids)) if ids else torch.empty(0, dtype=torch.int32)
        self.target_dates = pd.DatetimeIndex(np.concatenate(dates)) if dates else pd.DatetimeIndex([])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        start = self.valid_indices[index].item()
        target = self.target_indices[index].item()
        return self.data[start:target], self.targets[target]


def load_variance(database, estimator, asset='equity', ticker=None, limit_tickers=None,
                  limit_rows=None, include_intraday_log_return=False):
    if estimator not in ESTIMATORS or asset not in ('equity', 'crypto'):
        raise ValueError('invalid estimator or asset')
    if (limit_rows is not None and limit_rows < 1) or (limit_tickers is not None and limit_tickers < 1):
        raise ValueError('limits must be positive')
    table = f'{asset}_{ESTIMATORS[estimator]}_variance'
    if include_intraday_log_return:
        history = 'raw_history' if asset == 'equity' else 'crypto_daily_history'
        key = ('r.Ticker = v.Ticker AND r.Date = v.Date' if asset == 'equity'
               else 'r.symbol = v.Ticker AND substr(r.time, 1, 10) = v.Date')
        source = f'{table} v LEFT JOIN {history} r ON {key}'
        columns = 'v.Ticker, v.Date, v.Variance, r.Open AS Open, r.Close AS Close'
        ticker_column, date_column = 'v.Ticker', 'v.Date'
    else:
        source, columns, ticker_column, date_column = table, 'Ticker, Date, Variance', 'Ticker', 'Date'
    with closing(sqlite3.connect(database)) as conn:
        names = [row[0] for row in conn.execute(f'SELECT DISTINCT Ticker FROM {table} ORDER BY Ticker')]
        if ticker is not None:
            names = [ticker] if ticker in names else []
        if limit_tickers:
            names = names[:limit_tickers]
        frames = []
        for name in names:
            if limit_rows is None:
                frames.append(pd.read_sql_query(
                    f'SELECT {columns} FROM {source} WHERE {ticker_column} = ? ORDER BY {date_column}',
                    conn, params=(name,)))
            else:
                periods = ((None, None),) if asset == 'crypto' else (
                    (None, '2016-01-01'), ('2016-01-01', '2019-01-01'),
                    ('2019-01-01', '2026-01-01'))
                for start, end in periods:
                    clause = f' AND {date_column} >= ?' if start else ''
                    clause += f' AND {date_column} < ?' if end else ''
                    order = 'DESC' if asset == 'equity' and start is None else 'ASC'
                    params = (name, *((start,) if start else ()), *((end,) if end else ()), limit_rows)
                    query = (f'SELECT {columns} FROM {source} WHERE {ticker_column} = ?'
                             f'{clause} ORDER BY {date_column} {order} LIMIT ?')
                    frames.append(pd.read_sql_query(query, conn, params=params))
    if not frames:
        raise ValueError(f'no data in {table} for selected tickers')
    frame = pd.concat(frames, ignore_index=True).sort_values(['Ticker', 'Date']).reset_index(drop=True)
    if not np.isfinite(frame.Variance).all() or (frame.Variance <= 0).any():
        raise ValueError('invalid variance in source table')
    if include_intraday_log_return:
        prices = frame[['Open', 'Close']].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError('intraday log return needs matched finite positive Open and Close')
        frame['IntradayLogReturn'] = np.log(prices[:, 1] / prices[:, 0])
        if not np.isfinite(frame.IntradayLogReturn).all():
            raise ValueError('intraday log return must be finite')
        frame = frame.drop(columns=['Open', 'Close'])
    return frame


def series_by_ticker(frame, split='train'):
    start, end = SPLITS[split]
    if start:
        frame = frame[frame.Date >= start]
    if end:
        frame = frame[frame.Date < end]
    return {ticker: group.sort_values('Date').Variance.to_numpy(dtype=float)
            for ticker, group in frame.groupby('Ticker') if len(group)}


def forecast_floor(training):
    values = np.concatenate(list(training.values()))
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('fitting history needs finite positive variance')
    return float(values.min())


def floor_prediction(prediction, floor):
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('invalid forecast floor')
    if torch.is_tensor(prediction):
        return prediction.clamp_min(floor)
    return np.maximum(np.asarray(prediction, dtype=float), floor)


def qlike(actual, prediction):
    if torch.is_tensor(prediction):
        if not (torch.isfinite(actual).all() and torch.isfinite(prediction).all()
                and (actual > 0).all() and (prediction > 0).all()):
            raise ValueError('QLIKE requires finite positive variance')
        log_ratio = torch.log(actual) - torch.log(prediction)
        value = (torch.expm1(log_ratio) - log_ratio).mean()
        if not torch.isfinite(value):
            raise ValueError('QLIKE must be finite')
        return value
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if not (np.isfinite(actual).all() and np.isfinite(prediction).all()
            and (actual > 0).all() and (prediction > 0).all()):
        raise ValueError('QLIKE requires finite positive variance')
    log_ratio = np.log(actual) - np.log(prediction)
    value = float(np.mean(np.expm1(log_ratio) - log_ratio))
    if not math.isfinite(value):
        raise ValueError('QLIKE must be finite')
    return value


def variance_metrics(actual, prediction, tickers, training):
    actual, prediction, tickers = map(lambda x: np.asarray(x).reshape(-1),
                                       (actual, prediction, tickers))
    if not len(actual) or len(actual) != len(prediction) or len(actual) != len(tickers):
        raise ValueError('metric arrays must have equal nonzero lengths')
    scales = {}
    for ticker in set(tickers):
        if ticker not in training:
            raise ValueError(f'{ticker}: missing training history')
        history = np.asarray(training[ticker], dtype=float)
        if len(history) < 2 or not np.isfinite(history).all() or (history <= 0).any():
            raise ValueError('invalid MASE history')
        scales[ticker] = np.abs(np.diff(history)).mean()
        if scales[ticker] <= 0:
            raise ValueError('zero MASE scale')
    error = actual - prediction
    mse = np.mean(error ** 2)
    result = {'qlike': qlike(actual, prediction), 'mae': float(np.mean(np.abs(error))),
            'mase': float(np.mean([abs(e) / scales[t] for e, t in zip(error, tickers)])),
            'mse': float(mse), 'rmse': float(np.sqrt(mse))}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('variance metrics must be finite')
    return result


def add_common_args(parser, window=20):
    parser.add_argument('--database', required=True)
    parser.add_argument('--estimator', choices=ESTIMATORS, required=True)
    parser.add_argument('--scope', choices=('global', 'local'), required=True)
    parser.add_argument('--ticker')
    parser.add_argument('--run-name', default='default')
    parser.add_argument('--window-size', type=int, default=window)
    parser.add_argument('--limit-tickers', type=int)
    parser.add_argument('--limit-rows', type=int)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--crypto-test', action='store_true', help='score unseen crypto against the matching proxy')
    return parser


def prepare(args, include_intraday_log_return=False):
    if args.scope == 'local' and not args.ticker:
        raise ValueError('--ticker is required for local runs')
    if args.scope == 'global' and args.ticker:
        raise ValueError('--ticker applies only to local runs')
    frame = load_variance(args.database, args.estimator, ticker=args.ticker,
                          limit_tickers=args.limit_tickers, limit_rows=args.limit_rows,
                          include_intraday_log_return=include_intraday_log_return)
    training = series_by_ticker(frame)
    if not training:
        raise ValueError('no pre-2016 fitting history')
    return frame, training, forecast_floor(training)


def artifact_path(model, args):
    parts = [Path('inference/checkpoints'), model, args.estimator, args.scope]
    if args.scope == 'local':
        parts.append(args.ticker)
    parts.append(args.run_name)
    if any(not str(part) or str(part) in ('.', '..') or '/' in str(part) or '\\' in str(part)
           for part in parts[2:]):
        raise ValueError('invalid artifact path component')
    path = Path(*parts) / 'fit.pth'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_fit(path, fit):
    torch.save(fit, path)


def load_fit(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def report_counts(frame, window_size, consecutive_sessions=False, session_calendar=None):
    counts = {split: 0 for split in ('train', 'val', 'test')}
    calendar = (np.sort(pd.to_datetime(session_calendar).to_numpy()) if session_calendar is not None
                else np.sort(pd.to_datetime(frame.Date).unique())) if consecutive_sessions else None
    for _, group in frame.groupby('Ticker'):
        all_dates = pd.to_datetime(group.sort_values('Date').Date).to_numpy()
        dates = all_dates[window_size:]
        if consecutive_sessions:
            ranks = np.searchsorted(calendar, all_dates)
            dates = dates[ranks[window_size:] - ranks[:-window_size] == window_size]
        counts['train'] += int(np.sum(dates < np.datetime64('2016-01-01')))
        counts['val'] += int(np.sum((dates >= np.datetime64('2016-01-01')) &
                                    (dates < np.datetime64('2019-01-01'))))
        counts['test'] += int(np.sum((dates >= np.datetime64('2019-01-01')) &
                                     (dates < np.datetime64('2026-01-01'))))
    total = sum(counts.values())
    print(json.dumps({key: {'count': value, 'percent': round(100 * value / total, 2) if total else 0}
                      for key, value in counts.items()}))
    return counts


def wandb_run(args, model):
    if args.no_wandb:
        return None
    import wandb
    return wandb.init(project='timeseries-volatility', name=args.run_name,
                      config={**vars(args), 'model': model},
                      **({'id': args.wandb_id, 'resume': 'must'}
                         if getattr(args, 'wandb_id', None) else {}))
