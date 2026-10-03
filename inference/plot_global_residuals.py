"""Plot saved global test forecasts' raw and log GK variance residuals."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inference.evaluate_volatility_test import (NEURAL, STATISTICAL, EXPECTED, artifacts,
                                                RFSV_WINDOWS,
                                                load_raw_neural, scales_from_test,
                                                score_neural,
                                                score_statistical)
from models.training_blocks import TimeSeriesDataset


NAMES = {
    'MLP-20': 'mlp_20', 'MLP-80': 'mlp_80',
    'Base LSTM-20': 'base_lstm_20', 'Base LSTM-80': 'base_lstm_80',
    'SiLU-LSTM-20': 'silu_lstm_20', 'SiLU-LSTM-80': 'silu_lstm_80',
    'HARNet-20': 'harnet_20', 'HARNet-80': 'harnet_80',
    'AR(1)': 'ar1', 'HAR-20': 'har_20', 'HAR-80': 'har_80',
    'GARCH(1,1)': 'garch_11', 'SARIMA': 'sarima',
    'RFSV-20': 'rfsv_20', 'RFSV-80': 'rfsv_80',
}


def mean_variances_by_date(dates, actual, predicted):
    days, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    return (days, np.bincount(inverse, weights=actual) / counts,
            np.bincount(inverse, weights=predicted) / counts)


def plot_residuals(label, dates, actuals, predictions, output):
    dates = np.concatenate(dates).astype('datetime64[D]')
    actual = np.concatenate(actuals)
    predicted = np.concatenate(predictions)
    if (not len(dates) or len(dates) != len(actual) or len(actual) != len(predicted)
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or (actual <= 0).any() or (predicted <= 0).any()):
        raise ValueError(f'{label}: missing, misaligned, or nonpositive variances')
    output.mkdir(parents=True, exist_ok=True)
    days, mean_actual, mean_predicted = mean_variances_by_date(dates, actual, predicted)
    for name, residuals, xlabel in (
            ('raw', actual - predicted, 'Actual - predicted (daily unannualized adjusted GK variance)'),
            ('log', np.log(actual) - np.log(predicted), 'Log(actual variance) - log(predicted variance)')):
        low, high = np.quantile(residuals, [.0025, .9975])
        central = (residuals >= low) & (residuals <= high)
        counts, edges = np.histogram(residuals, bins=120)
        central_counts, central_edges = np.histogram(residuals[central], bins=120)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        for ax, values, bins, title in zip(axes, (counts, central_counts),
                                           (edges, central_edges), ('All residuals', 'Central 99.5%')):
            ax.stairs(values, bins, fill=True, color='#236c93')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Forecast count')
            ax.grid(axis='y', alpha=.25)
        fig.suptitle(f'{label} | {name.title()} test residuals | N = {len(residuals):,}')
        fig.tight_layout()
        fig.savefig(output / f'histogram_{name}.png', dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(dates[central], residuals[central], ',', color='#236c93', alpha=.2, rasterized=True)
        ax.axhline(0, color='#a03d3d', linewidth=.8)
        ax.set_xlabel('Test target date')
        ax.set_ylabel(xlabel)
        ax.set_title(f'{label} | Dated {name} residuals | Central 99.5% | N = {int(central.sum()):,}')
        if np.unique(dates[central]).size == 1:
            day = dates[central][0]
            ax.set_xlim(day - np.timedelta64(3, 'D'), day + np.timedelta64(3, 'D'))
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output / f'timeline_{name}.png', dpi=150)
        plt.close(fig)
        means = (mean_actual - mean_predicted if name == 'raw' else
                 np.log(mean_actual) - np.log(mean_predicted))
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(days, means, color='#236c93', linewidth=.9,
                marker='o' if len(days) == 1 else None)
        ax.axhline(0, color='#a03d3d', linewidth=.8)
        ax.set_xlabel('Test target date')
        ax.set_ylabel('Mean(actual) - mean(predicted)\nGK variance' if name == 'raw'
                      else 'Log(mean actual) - log(mean predicted)\ndimensionless')
        ax.set_title(f'{label} | Daily {"mean raw residual" if name == "raw" else "log residual of ticker means"} | '
                     f'{len(days):,} date{"s" if len(days) != 1 else ""}')
        if len(days) == 1:
            ax.set_xlim(days[0] - np.timedelta64(3, 'D'), days[0] + np.timedelta64(3, 'D'))
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output / f'timeline_mean_{name}.png', dpi=150)
        plt.close(fig)
    return len(dates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default='D:/DBs/timeseries_analysis/history_coverage.db')
    parser.add_argument('--output', type=Path, default=Path('imgs/eval/global_eval/residuals'))
    parser.add_argument('--batch-size', type=int, default=2048)
    args = parser.parse_args()
    frame, _, floor = load_raw_neural(args.database)
    frame = frame.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    fits = artifacts(args.database, frame, floor)
    print(json.dumps({'preflight': '15 artifacts verified', 'floor': floor}), flush=True)

    dates, actuals, predictions = [], [], []
    def emit(batch_dates, batch_actual, batch_predicted):
        dates.append(batch_dates)
        actuals.append(batch_actual)
        predictions.append(batch_predicted)

    for window in (20, 80):
        data = TimeSeriesDataset(frame, window, split='test', valid_column='Valid')
        if len(data) != EXPECTED[window]['test']:
            raise ValueError(f'window {window}: unexpected test count')
        scales = scales_from_test(frame, data)
        ids = data.ticker_ids.numpy()
        names = data.ticker_names
        eligible = np.array([np.isfinite(scales.get(name, np.nan)) and scales.get(name, np.nan) > 0
                             for name in names])
        expected = int(eligible[ids].sum())
        for label, (_, fit, fitted_window) in fits.items():
            if fitted_window != window:
                continue
            dates, actuals, predictions = [], [], []
            result = (score_neural(NEURAL[label][0], fit, frame, data, scales, floor,
                                   args.batch_size, emit=emit) if label in NEURAL else
                      score_statistical('rfsv' if label in RFSV_WINDOWS else STATISTICAL[label][0],
                                        fit, frame, data, scales, floor, emit=emit))
            if result['N'] != expected:
                raise ValueError(f'{label}: count differs from shared test population')
            count = plot_residuals(label, dates, actuals, predictions, args.output / NAMES[label])
            if count != expected:
                raise ValueError(f'{label}: residual count differs from score count')
            print(json.dumps({'model': label, 'residuals': count}), flush=True)
        del data


if __name__ == '__main__':
    main()
