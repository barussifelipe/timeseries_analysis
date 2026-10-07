"""Plot the global LSTM test-date means against preceding stock references."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.plot_data_properties import WINDOW
from inference.evaluate_volatility_test import (EXPECTED, NEURAL, artifacts,
                                                scales_from_test, score_neural)
from models.support_scripts.raw_neural_data import load_raw_neural
from models.training_blocks import TimeSeriesDataset


def target_references(frame, data, scales, predictions=None):
    """Align strictly preceding 252-valid-observation means to scored targets."""
    valid = frame.loc[frame.Valid, ['Ticker', 'Variance']].copy()
    ids = data.ticker_ids.numpy()
    eligible = np.array([np.isfinite(scales.get(name, np.nan)) and scales.get(name, np.nan) > 0
                         for name in data.ticker_names])
    targets = data.target_indices.numpy()[eligible[ids]]
    if predictions is not None:
        predictions = np.asarray(predictions, dtype=float)
        if len(predictions) != len(targets) or not np.isfinite(predictions).all() or (predictions <= 0).any():
            raise ValueError('invalid predicted reference values')
        valid.loc[targets, 'Variance'] = predictions
    valid['reference'] = valid.groupby('Ticker', observed=True).Variance.transform(
        lambda values: values.shift().rolling(WINDOW, min_periods=WINDOW).mean())
    selected = valid.reindex(targets)
    if (selected.reference.notna() & ((~np.isfinite(selected.reference))
                                      | (selected.reference <= 0))).any():
        raise ValueError('nonpositive or nonfinite complete reference')
    return selected.reference.to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default='D:/DBs/timeseries_analysis/history_coverage.db')
    parser.add_argument('--daily-means', type=Path,
                        default=Path('imgs/data_properties/global_base_lstm80_daily_means.csv'))
    parser.add_argument('--output', type=Path, default=Path('imgs/data_properties'))
    parser.add_argument('--predicted-reference-preview', action='store_true')
    args = parser.parse_args()

    frame, _, floor = load_raw_neural(args.database)
    frame = frame.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    data = TimeSeriesDataset(frame, 80, split='test', valid_column='Valid')
    if len(data) != EXPECTED[80]['test']:
        raise ValueError('unexpected global 80-window test count')
    scales = scales_from_test(frame, data)
    fit = artifacts(args.database, frame, floor)['Base LSTM-80'][1]
    chunks = []

    def emit(dates, actual, predicted):
        chunks.append(pd.DataFrame({'date': dates, 'actual': actual,
                                    'predicted': predicted}))

    scored = score_neural(NEURAL['Base LSTM-80'][0], fit, frame, data, scales,
                          floor, 8192, emit=emit)
    targets = pd.concat(chunks, ignore_index=True)
    if scored['N'] != len(targets):
        raise ValueError('forecast and reference target order mismatch')
    targets['reference'] = target_references(
        frame, data, scales,
        targets.predicted.to_numpy() if args.predicted_reference_preview else None)
    full = targets.groupby('date', sort=True)[['actual', 'predicted']].mean()
    full['count'] = targets.groupby('date', sort=True).size()
    saved = pd.read_csv(args.daily_means, parse_dates=['date']).set_index('date')
    if (len(full) != 1760 or not full.index.equals(saved.index)
            or not np.array_equal(full['count'], saved['count'])
            or not np.allclose(full[['actual', 'predicted']], saved[['actual', 'predicted']],
                               rtol=1e-9, atol=1e-12)):
        raise ValueError('saved daily means do not match the eligible global test targets')
    complete = targets.dropna(subset=['reference'])
    grouped = complete.groupby('date', sort=True)
    daily = grouped[['actual', 'predicted', 'reference']].mean()
    daily['count'] = grouped.size()
    if (not daily.index.equals(saved.index) or not np.isfinite(daily.to_numpy()).all()
            or (daily[['actual', 'predicted', 'reference']] <= 0).any().any()):
        raise ValueError('incomplete or invalid global rolling series')
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = '_predicted_reference_preview' if args.predicted_reference_preview else ''
    daily.to_csv(args.output / f'global_base_lstm80_rolling_daily_means{suffix}.csv')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily.index, np.log(daily.actual / daily.reference), color='#236c93',
            lw=.8, label='Actual')
    ax.plot(daily.index, np.log(daily.predicted / daily.reference), color='#bc4b2f',
            lw=.8, label='Base LSTM-80 Global')
    ax.axhline(0, color='#555555', lw=.6)
    ax.set(xlabel='Test target date', ylabel='Log variance / trailing reference',
           title=('Global equity mean variance relative to preceding 252-observation '
                  + ('forecast-updated mean' if args.predicted_reference_preview else 'mean')))
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / f'global_base_lstm80_mean_reversion{suffix}.png', dpi=150)
    plt.close(fig)
    print({'test_dates': len(daily), 'plotted_forecasts': int(daily['count'].sum()),
           'excluded_short_references': int(len(targets) - len(complete)),
           'eligible_stocks_per_date': (int(daily['count'].min()), int(daily['count'].max()))})


if __name__ == '__main__':
    main()
