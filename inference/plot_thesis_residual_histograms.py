"""Rebuild the ten thesis residual histogram pairs from saved MCS forecasts."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from inference.plot_global_residuals import NAMES, plot_residual_histogram


MODELS = (
    ('Base LSTM-20', 'global'), ('Base LSTM-80', 'global'),
    ('Base LSTM-80', 'local'), ('HARNet-20', 'global'),
    ('HARNet-80', 'global'), ('HARNet-80', 'local'),
    ('MLP-20', 'global'), ('MLP-80', 'global'),
    ('RFSV-20', 'local'), ('RFSV-80', 'local'),
)


def main():
    root = Path('imgs/eval')
    for window in (20, 80):
        wanted = {f'{model} {scope.upper()}' for model, scope in MODELS
                  if model.endswith(f'-{window}')}
        rows = defaultdict(list)
        with (root / 'mcs' / f'{window}_size' / 'forecasts.csv').open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if row['candidate'] in wanted:
                    rows[row['candidate']].append(row)
        if set(rows) != wanted:
            raise ValueError(f'window {window}: missing thesis forecasts')
        for model, scope in MODELS:
            if not model.endswith(f'-{window}'):
                continue
            label = f'{model} {scope.upper()}'
            selected = rows[label]
            keys = {(row['ticker'], row['date']) for row in selected}
            if len(selected) != 8800 or len(keys) != 8800:
                raise ValueError(f'{label}: expected 8,800 unique ticker-date forecasts')
            actual = np.array([float(row['actual']) for row in selected])
            predicted = np.array([float(row['predicted']) for row in selected])
            if (not np.isfinite(actual).all() or not np.isfinite(predicted).all()
                    or (actual <= 0).any() or (predicted <= 0).any()):
                raise ValueError(f'{label}: invalid variance')
            destination = root / 'local_eval' / 'residuals' / NAMES[model] / scope
            for name, residuals, xlabel in (
                    ('raw', actual - predicted, 'Actual - predicted (daily unannualized adjusted GK variance)'),
                    ('log', np.log(actual) - np.log(predicted),
                     'Log(actual variance) - log(predicted variance)')):
                plot_residual_histogram(label, name, residuals, xlabel,
                                        destination / f'histogram_{name}.png', full_only=True)
            print(f'{label}: {len(selected):,} residuals')


if __name__ == '__main__':
    main()
