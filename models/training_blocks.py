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
    """One target date; actual is unchanged, prediction is in variance units and floored."""

    ticker: str
    target_date: str
    actual_variance: float
    predicted_variance: float


class TimeSeriesDataset(Dataset):
    """Next-observation windows, stored as one tensor sorted by ticker and date.

    For target row i, input rows are [i - window_size, i) from the same ticker.
    The split filters target dates only, so validation and test windows can use
    earlier observed history. Variance targets must be finite and positive.
    ``positive_target=False`` preserves the older returns reference behavior.
    """

    def __init__(self, dataframe, window_size=30, feature_columns=('Variance',),
                 target_column='Variance', split='train', positive_target=True):
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
        numbers = frame[list(set(self.feature_columns) | {target_column})].to_numpy(dtype=float)
        if not np.isfinite(numbers).all() or (positive_target and (frame[target_column].to_numpy(dtype=float) <= 0).any()):
            raise ValueError('variance data must be finite and targets positive')
        self.data = torch.tensor(frame[list(self.feature_columns)].to_numpy(dtype=np.float32))
        self.targets = torch.tensor(frame[target_column].to_numpy(dtype=np.float32)).unsqueeze(1)
        if (not torch.isfinite(self.data).all() or not torch.isfinite(self.targets).all()
                or (positive_target and (self.targets <= 0).any())):
            raise ValueError('values cannot be represented as finite positive float32 variance')
        self.ticker_names = list(frame['Ticker'].drop_duplicates())
        indices, ids, dates = [], [], []
        start, end = SPLITS[split]
        offset = 0
        for ticker_id, (ticker, group) in enumerate(frame.groupby('Ticker', sort=False)):
            positions = np.arange(window_size, len(group))
            group_dates = group['Date'].to_numpy()[positions]
            mask = np.ones(len(positions), dtype=bool)
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
                  limit_rows=None):
    if estimator not in ESTIMATORS or asset not in ('equity', 'crypto'):
        raise ValueError('invalid estimator or asset')
    if (limit_rows is not None and limit_rows < 1) or (limit_tickers is not None and limit_tickers < 1):
        raise ValueError('limits must be positive')
    table = f'{asset}_{ESTIMATORS[estimator]}_variance'
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
                    f'SELECT Ticker, Date, Variance FROM {table} WHERE Ticker = ? ORDER BY Date',
                    conn, params=(name,)))
            else:
                periods = ((None, None),) if asset == 'crypto' else (
                    (None, '2016-01-01'), ('2016-01-01', '2019-01-01'),
                    ('2019-01-01', '2026-01-01'))
                for start, end in periods:
                    clause = ' AND Date >= ?' if start else ''
                    clause += ' AND Date < ?' if end else ''
                    order = 'DESC' if asset == 'equity' and start is None else 'ASC'
                    params = (name, *((start,) if start else ()), *((end,) if end else ()), limit_rows)
                    query = (f'SELECT Ticker, Date, Variance FROM {table} WHERE Ticker = ?'
                             f'{clause} ORDER BY Date {order} LIMIT ?')
                    frames.append(pd.read_sql_query(query, conn, params=params))
    if not frames:
        raise ValueError(f'no data in {table} for selected tickers')
    frame = pd.concat(frames, ignore_index=True).sort_values(['Ticker', 'Date']).reset_index(drop=True)
    if not np.isfinite(frame.Variance).all() or (frame.Variance <= 0).any():
        raise ValueError('invalid variance in source table')
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
        ratio = actual / prediction
        return (ratio - torch.log(ratio) - 1).mean()
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


def prepare(args):
    if args.scope == 'local' and not args.ticker:
        raise ValueError('--ticker is required for local runs')
    if args.scope == 'global' and args.ticker:
        raise ValueError('--ticker applies only to local runs')
    frame = load_variance(args.database, args.estimator, ticker=args.ticker,
                          limit_tickers=args.limit_tickers, limit_rows=args.limit_rows)
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


def report_counts(frame, window_size):
    counts = {split: 0 for split in ('train', 'val', 'test')}
    for _, group in frame.groupby('Ticker'):
        dates = pd.to_datetime(group.sort_values('Date').Date).to_numpy()[window_size:]
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
                      config={**vars(args), 'model': model})
