"""Simulate adjusted closes from saved one-day variance forecasts."""

import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.plot_gk_data import LOCAL
from inference.evaluate_volatility_test import (EXPECTED, NEURAL, artifacts,
                                                scales_from_test, score_neural)
from models.support_scripts.raw_neural_data import load_raw_neural
from models.training_blocks import TimeSeriesDataset


PLOTS = (
    ('local_mlp80_price_reconstruction', 'Five-stock adjusted Close simulation | MLP-80 Global'),
    ('global_base_lstm80_price_reconstruction', 'Global equity adjusted Close simulation (excluding WHLR) | Base LSTM-80 Global'),
)
MLP_GLOBAL = ('global_mlp80_price_reconstruction_gross_return',
              'Global equity adjusted Close simulation (excluding WHLR) | MLP-80 Global | gross Open-to-Close return')


def preceding_drift(frame, targets, window=252):
    """Mean of the preceding valid adjusted intraday returns per ticker."""
    valid = frame.loc[frame.Valid, ['Ticker', 'IntradayLogReturn']].copy()
    valid['drift'] = valid.groupby('Ticker', observed=True).IntradayLogReturn.transform(
        lambda x: x.shift().rolling(window, min_periods=window).mean())
    return valid.reindex(targets).drift.to_numpy(dtype=float)


def adjusted_prices(database, frame, targets):
    """Read exact test-target adjusted prices, rejecting missing or invalid joins."""
    opened = np.full(len(targets), np.nan)
    closed = np.full(len(targets), np.nan)
    ids = frame.Ticker.cat.codes.to_numpy()[targets]
    dates = frame.Date.to_numpy()[targets]
    with sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True) as conn:
        for ticker_id, ticker in enumerate(frame.Ticker.cat.categories):
            positions = np.arange(np.searchsorted(ids, ticker_id, side='left'),
                                  np.searchsorted(ids, ticker_id, side='right'))
            if not len(positions):
                continue
            source = pd.read_sql_query('''SELECT Date, Open, Close FROM raw_history
                WHERE Ticker = ? AND Date >= '2019-01-01' AND Date < '2026-01-01'
                ORDER BY Date''', conn, params=(ticker,))
            source.Date = pd.to_datetime(source.Date)
            if source.Date.duplicated().any():
                raise ValueError(f'{ticker}: duplicate adjusted price date')
            matched = source.set_index('Date').reindex(dates[positions])
            opened[positions] = matched.Open.to_numpy(dtype=float)
            closed[positions] = matched.Close.to_numpy(dtype=float)
    if (not np.isfinite(opened).all() or not np.isfinite(closed).all()
            or (opened <= 0).any() or (closed <= 0).any()):
        raise ValueError('missing or nonpositive adjusted target Open/Close')
    return opened, closed


def simulate(opened, variance, drift, shock):
    if (not np.isfinite([opened, variance, drift, shock]).all()
            or (np.asarray(opened) <= 0).any() or (np.asarray(variance) <= 0).any()):
        raise ValueError('simulation needs finite inputs and positive Open/variance')
    close = opened * np.exp(drift + np.sqrt(variance) * shock)
    if not np.isfinite(close).all() or (np.asarray(close) <= 0).any():
        raise ValueError('simulated Close is invalid')
    return close


def daily(values):
    grouped = values.groupby('date', sort=True)
    result = grouped[['actual_close', 'simulated_close']].mean()
    result['count'] = grouped.size()
    if not np.isfinite(result.to_numpy()).all() or (result[['actual_close', 'simulated_close']] <= 0).any().any():
        raise ValueError('invalid date-level mean close')
    return result


def log_daily_means(values):
    """Log the already averaged daily prices, keeping the same date populations."""
    result = values[['actual_close', 'simulated_close']].copy()
    if not np.isfinite(result.to_numpy()).all() or (result <= 0).any().any():
        raise ValueError('log daily means need finite positive prices')
    result = np.log(result).rename(columns={'actual_close': 'log_actual_mean_close',
                                            'simulated_close': 'log_simulated_mean_close'})
    result['count'] = values['count']
    return result


def daily_gross_returns(values):
    """Average each stock's adjusted Close/Open on the same eligible dates."""
    opened = values['open'].to_numpy(dtype=float)
    closes = values[['actual_close', 'simulated_close']].to_numpy(dtype=float)
    if (not np.isfinite(opened).all() or (opened <= 0).any()
            or not np.isfinite(closes).all() or (closes <= 0).any()):
        raise ValueError('gross returns need finite positive adjusted prices')
    ratios = closes / opened[:, None]
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError('invalid adjusted Close/Open ratio')
    frame = pd.DataFrame({'date': values['date'], 'actual_gross_return': ratios[:, 0],
                          'simulated_gross_return': ratios[:, 1]})
    grouped = frame.groupby('date', sort=True)
    result = grouped[['actual_gross_return', 'simulated_gross_return']].mean()
    result['count'] = grouped.size()
    return result


def plot(values, path, title, log_y=False, log_values=False, gross_returns=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    actual = 'actual_gross_return' if gross_returns else 'log_actual_mean_close' if log_values else 'actual_close'
    simulated = ('simulated_gross_return' if gross_returns else
                 'log_simulated_mean_close' if log_values else 'simulated_close')
    ax.plot(values.index, values[actual], color='#236c93', lw=.8,
            label=('Actual adjusted Close/Open' if gross_returns else
                   'Log actual mean adjusted Close' if log_values else 'Actual adjusted Close'))
    ax.plot(values.index, values[simulated], color='#bc4b2f', lw=.8,
            label=('Simulated adjusted Close/Open' if gross_returns else
                   'Log simulated mean adjusted Close' if log_values else
                   'Simulated adjusted Close (one shock per stock-date)'))
    if log_y:
        ax.set_yscale('log')
    if gross_returns:
        ax.axhline(1, color='#555555', lw=.6)
    ax.set(xlabel='Test target date',
           ylabel=('Mean adjusted Close/Open (gross return)' if gross_returns else
                   'Log(mean adjusted Close / USD)' if log_values else
                   'Mean adjusted Close (USD per stock' + (', log scale)' if log_y else ')')),
           title=title)
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default='D:/DBs/timeseries_analysis/history_coverage.db')
    parser.add_argument('--forecasts', type=Path, default=Path('imgs/eval/mcs/80_size/forecasts.csv'))
    parser.add_argument('--output', type=Path, default=Path('imgs/data_properties'))
    parser.add_argument('--show', action='store_true',
                        help='open price, log-mean, and gross-return plots from saved CSVs')
    args = parser.parse_args()

    if args.show:
        plt.switch_backend('TkAgg')
        for name, title in PLOTS:
            for suffix in ('', '_log', '_gross_return'):
                values = pd.read_csv(args.output / f'{name}{suffix}.csv', parse_dates=['date']).set_index('date')
                plot(values, None, title + (' | log of daily mean' if suffix == '_log' else
                                            ' | gross Open-to-Close return' if suffix else ''),
                     log_y=name.startswith('global_') and not suffix,
                     log_values=suffix == '_log', gross_returns=suffix == '_gross_return')
        values = pd.read_csv(args.output / f'{MLP_GLOBAL[0]}.csv', parse_dates=['date']).set_index('date')
        plot(values, None, MLP_GLOBAL[1], gross_returns=True)
        plt.show()
        return

    frame, _, floor = load_raw_neural(args.database)
    frame = frame.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    fits = artifacts(args.database, frame, floor)
    fit = fits['Base LSTM-80'][1]
    data = TimeSeriesDataset(frame, 80, split='test', valid_column='Valid')
    if len(data) != EXPECTED[80]['test']:
        raise ValueError('unexpected global window-80 test count')
    scales = scales_from_test(frame, data)
    ids = data.ticker_ids.numpy()
    eligible = np.array([np.isfinite(scales.get(name, np.nan)) and scales.get(name, np.nan) > 0
                         for name in data.ticker_names])
    targets = data.target_indices.numpy()[eligible[ids]]
    tickers = frame.Ticker.to_numpy()[targets]
    dates = frame.Date.to_numpy()[targets]
    drift = preceding_drift(frame, targets)
    opened, closed = adjusted_prices(args.database, frame, targets)
    shocks = np.random.default_rng(0).standard_normal(len(targets))
    usable = np.isfinite(drift) & (tickers != 'WHLR')
    if not np.isfinite(shocks).all():
        raise ValueError('nonfinite sampled shock')
    chunks = []
    offset = 0

    def emit(batch_dates, actual, predicted):
        nonlocal offset
        end = offset + len(batch_dates)
        sl = slice(offset, end)
        if (not np.array_equal(batch_dates, dates[sl])
                or not np.array_equal(actual, frame.Variance.to_numpy()[targets[sl]])):
            raise ValueError('global forecast target alignment failed')
        mask = usable[sl]
        chunks.append(pd.DataFrame({'date': batch_dates[mask], 'open': opened[sl][mask],
                                    'actual_close': closed[sl][mask],
                                    'simulated_close': simulate(opened[sl][mask], predicted[mask],
                                                                drift[sl][mask], shocks[sl][mask])}))
        offset = end

    scored = score_neural(NEURAL['Base LSTM-80'][0], fit, frame, data, scales,
                          floor, 8192, emit=emit)
    if offset != len(targets) or scored['N'] != offset:
        raise ValueError('global forecast count mismatch')
    global_values = pd.concat(chunks, ignore_index=True)
    global_daily = daily(global_values)
    global_gross = daily_gross_returns(global_values)
    chunks = []
    offset = 0
    mlp_scored = score_neural(NEURAL['MLP-80'][0], fits['MLP-80'][1], frame, data,
                              scales, floor, 8192, emit=emit)
    if offset != len(targets) or mlp_scored['N'] != offset:
        raise ValueError('global MLP forecast count mismatch')
    mlp_gross = daily_gross_returns(pd.concat(chunks, ignore_index=True))
    if not mlp_gross['count'].equals(global_gross['count']):
        raise ValueError('global MLP and LSTM plotted populations differ')

    local_mask = np.isin(tickers, list(LOCAL))
    local_keys = pd.DataFrame({'ticker': tickers[local_mask],
                               'date': dates[local_mask],
                               'actual_variance': frame.Variance.to_numpy()[targets[local_mask]],
                               'open': opened[local_mask],
                               'actual_close': closed[local_mask],
                               'drift': drift[local_mask],
                               'shock': shocks[local_mask]})
    saved = pd.read_csv(args.forecasts)
    saved = saved.loc[saved.candidate == 'MLP-80 GLOBAL', ['ticker', 'date', 'actual', 'predicted']].copy()
    saved.date = pd.to_datetime(saved.date)
    if (len(saved) != 8800 or saved.duplicated(['ticker', 'date']).any()
            or set(saved.ticker) != LOCAL):
        raise ValueError('expected five unique saved MLP-80 Global forecast series')
    local = local_keys.merge(saved, on=['ticker', 'date'], how='outer', validate='one_to_one', indicator=True)
    if (len(local) != len(saved) or (local._merge != 'both').any()
            or not np.allclose(local.actual, local.actual_variance, rtol=1e-10, atol=1e-13)
            or not np.isfinite(local.predicted).all() or (local.predicted <= 0).any()):
        raise ValueError('local forecast keys, target variance, or prediction mismatch')
    complete = local.loc[np.isfinite(local.drift)].copy()
    complete['simulated_close'] = simulate(complete.open.to_numpy(), complete.predicted.to_numpy(),
                                            complete.drift.to_numpy(), complete.shock.to_numpy())
    local_daily = daily(complete)
    local_gross = daily_gross_returns(complete)
    args.output.mkdir(parents=True, exist_ok=True)
    for (name, title), values, gross in zip(PLOTS, (local_daily, global_daily), (local_gross, global_gross)):
        values.to_csv(args.output / f'{name}.csv')
        plot(values, args.output / f'{name}.png', title, log_y=name.startswith('global_'))
        logged = log_daily_means(values)
        logged.to_csv(args.output / f'{name}_log.csv')
        plot(logged, args.output / f'{name}_log.png', title + ' | log of daily mean', log_values=True)
        gross.to_csv(args.output / f'{name}_gross_return.csv')
        plot(gross, args.output / f'{name}_gross_return.png',
             title + ' | gross Open-to-Close return', gross_returns=True)
    mlp_gross.to_csv(args.output / f'{MLP_GLOBAL[0]}.csv')
    plot(mlp_gross, args.output / f'{MLP_GLOBAL[0]}.png', MLP_GLOBAL[1], gross_returns=True)
    print({'local': {'forecasts': len(local), 'excluded_short_history': len(local) - len(complete),
                     'plotted': len(complete), 'dates': len(local_daily)},
           'global': {'forecasts': len(targets), 'excluded_short_history': int((~np.isfinite(drift)).sum()),
                      'excluded_whlr': int((np.isfinite(drift) & (tickers == 'WHLR')).sum()),
                      'plotted': int(usable.sum()), 'dates': len(global_daily),
                      'stocks_per_date': (int(global_daily['count'].min()), int(global_daily['count'].max()))}})


if __name__ == '__main__':
    main()
