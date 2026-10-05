"""Plot observed adjusted GK variance for the forecasting cohort and five stocks."""

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.support_scripts.raw_neural_data import raw_frame


LOCAL = {'NVDA', 'AAPL', 'NFLX', 'GOOG', 'AMZN'}
CUTOFF = np.datetime64('2016-01-01')


def observations(conn):
    names = [row[0] for row in conn.execute("""SELECT Ticker FROM raw_history GROUP BY Ticker
        HAVING SUM(Date < '2016-01-01') > 80 AND SUM(Date = '2025-12-31') > 0
        ORDER BY Ticker""")]
    if len(names) != 2714 or not LOCAL <= set(names):
        raise ValueError('forecasting cohort or five-stock membership changed')
    parts = {'global': ([], []), 'local': ([], [])}
    counts = {scope: {'raw': 0, 'invalid': 0, 'zeros': 0,
                       'train_raw': 0, 'train_invalid': 0, 'train_zeros': 0}
              for scope in parts}
    floors = {}
    for name in names:
        source = pd.read_sql_query("""SELECT Ticker, Date, Open, High, Low, Close, Volume
            FROM raw_history WHERE Ticker = ? AND Date < '2026-01-01' ORDER BY Date""",
            conn, params=(name,))
        frame = raw_frame(source)
        dates = source.Date.to_numpy(dtype='datetime64[D]')
        valid = frame.Valid.to_numpy(dtype=bool)
        values = frame.RawVariance.to_numpy(dtype=float)
        train = dates < CUTOFF
        positive_train = values[valid & train & (values > 0)]
        if name in LOCAL:
            if not len(positive_train):
                raise ValueError(f'{name}: no positive training GK variance')
            floors[name] = float(positive_train.min())
        for scope in ('global', 'local') if name in LOCAL else ('global',):
            c = counts[scope]
            c['raw'] += len(source)
            c['invalid'] += int((~valid).sum())
            c['zeros'] += int((valid & (values == 0)).sum())
            c['train_raw'] += int(train.sum())
            c['train_invalid'] += int((train & ~valid).sum())
            c['train_zeros'] += int((train & valid & (values == 0)).sum())
            parts[scope][0].append(dates[valid])
            parts[scope][1].append(np.where(values[valid] == 0, floors[name], values[valid])
                                      if scope == 'local' else values[valid])
        if len(positive_train):
            floors.setdefault('global', float('inf'))
            floors['global'] = min(floors['global'], float(positive_train.min()))
    result = {}
    for scope, (day_parts, value_parts) in parts.items():
        days = np.concatenate(day_parts)
        variance = np.concatenate(value_parts)
        if scope == 'global':
            variance[variance == 0] = floors['global']
        if not np.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError(f'{scope}: invalid logarithm input')
        result[scope] = (days, variance)
        counts[scope]['plotted'] = len(days)
        counts[scope]['train_plotted'] = int((days < CUTOFF).sum())
    return result, counts, floors, names


def plot(days, variance, output, title):
    output.mkdir(parents=True, exist_ok=True)
    unique, daily = daily_mean(days, variance)
    for scale, series, ylabel in (('raw', daily, 'Daily unannualized GK variance'),
                                 ('log', np.log(daily), 'Log daily GK variance')):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(unique, series, color='#236c93', linewidth=.8)
        ax.set(xlabel='Date', ylabel=ylabel, title=f'{title}: {scale} daily cross-ticker mean')
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output / f'timeline_{scale}.png', dpi=150)
        plt.close(fig)
    for scale, values, xlabel in (('raw', variance, 'Daily unannualized GK variance'),
                                 ('log', np.log(variance), 'Log daily GK variance')):
        mean, std = values.mean(), values.std(ddof=0)
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f'{title}: invalid normal fit')
        lo, hi = np.quantile(values, [.0025, .9975])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, shown, label in zip(axes, (values, values[(values >= lo) & (values <= hi)]),
                                    ('All observations', 'Central 99.5%')):
            counts, edges = np.histogram(shown, bins=120)
            ax.stairs(counts, edges, fill=True, color='#236c93', label='Observed')
            x = np.linspace(edges[0], edges[-1], 1000)
            normal = np.exp(-.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            ax.plot(x, len(values) * (edges[1] - edges[0]) * normal,
                    color='#bc4b2f', linewidth=2, label='Normal fit to all observations')
            ax.set(xlabel=xlabel, ylabel='Observation count', title=label)
            ax.grid(axis='y', alpha=.25)
            ax.legend()
        fig.suptitle(f'{title}: {scale} GK variance | N = {len(values):,}\n'
                     f'Normal fit: mean = {mean:.4g}, standard deviation = {std:.4g}')
        fig.tight_layout()
        fig.savefig(output / f'histogram_{scale}.png', dpi=150)
        plt.close(fig)


def daily_mean(days, variance):
    unique, inverse, n = np.unique(days, return_inverse=True, return_counts=True)
    return unique, np.bincount(inverse, weights=variance) / n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('D:/DBs/timeseries_analysis/history_coverage.db'))
    parser.add_argument('--output', type=Path, default=Path('imgs/gk_data'))
    args = parser.parse_args()
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        populations, counts, floors, names = observations(conn)
    for scope, (days, variance) in populations.items():
        for period, keep in (('full', np.ones(len(days), dtype=bool)),
                             ('train', days < CUTOFF)):
            plot(days[keep], variance[keep], args.output / scope / period,
                 f'{scope.title()} {period} cohort')
    metadata = {'cohort_tickers': len(names), 'local_tickers': sorted(LOCAL),
                'end_date': '2025-12-31', 'train_end_exclusive': '2016-01-01',
                'global_floor': floors['global'],
                'local_floors': {name: floors[name] for name in sorted(LOCAL)},
                'counts': counts}
    (args.output / 'counts.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
