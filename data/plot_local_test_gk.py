"""Plot the five-stock GK targets shared by the local test forecasts."""

import argparse
from pathlib import Path

import numpy as np

from data.plot_gk_data import plot
from inference.run_local_base_lstm import DATABASE, TICKERS
from models.support_scripts.raw_neural_data import load_raw_neural
from models.training_blocks import TimeSeriesDataset


def test_targets(database):
    dates, values = [], []
    for ticker in TICKERS:
        frame, _, _ = load_raw_neural(database, ticker=ticker)
        frame = frame.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        datasets = [TimeSeriesDataset(frame, window, split='test', valid_column='Valid')
                    for window in (20, 80)]
        first, second = datasets
        if (len(first) != 1760 or len(second) != 1760
                or not first.target_dates.equals(second.target_dates)
                or not np.array_equal(first.targets[first.target_indices].numpy(),
                                      second.targets[second.target_indices].numpy())):
            raise ValueError(f'{ticker}: test targets differ between forecast windows')
        dates.append(first.target_dates.to_numpy(dtype='datetime64[D]'))
        values.append(first.targets[first.target_indices].numpy().ravel().astype(float))
    days, variance = np.concatenate(dates), np.concatenate(values)
    if (len(days) != 8800 or np.unique(days).size != 1760
            or not np.isfinite(variance).all() or (variance <= 0).any()):
        raise ValueError('invalid five-stock test targets')
    return days, variance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default=DATABASE)
    parser.add_argument('--output', type=Path, default=Path('imgs/eval/local_eval/test_data'))
    args = parser.parse_args()
    days, variance = test_targets(args.database)
    plot(days, variance, args.output, 'Five-stock test targets')
    print(f'Plotted {len(variance):,} targets on {np.unique(days).size:,} dates')


if __name__ == '__main__':
    main()
