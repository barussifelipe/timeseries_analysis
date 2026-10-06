"""Plot full-cohort test return/variance curves from the saved global LSTM."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.plot_data_properties import plot_return_curve
from inference.evaluate_volatility_test import (EXPECTED, NEURAL, ROOT, artifacts,
                                                scales_from_test, score_neural)
from models.support_scripts.raw_neural_data import load_raw_neural
from models.training_blocks import TimeSeriesDataset


LABEL = 'Base LSTM-80'


def aggregate_dates(dates, actual, predicted, preceding):
    values = pd.DataFrame({'date': dates, 'actual': actual,
                           'predicted': predicted, 'preceding_return': preceding})
    grouped = values.groupby('date', sort=True)
    means = grouped[['actual', 'predicted', 'preceding_return']].mean()
    means['count'] = grouped.size()
    return means


def daily_means(frame, data, scale_by_ticker, fit, floor, batch_size):
    """Aggregate the evaluator's eligible ticker forecasts on each target date."""
    ids = data.ticker_ids.numpy()
    eligible = np.array([np.isfinite(scale_by_ticker.get(name, np.nan))
                         and scale_by_ticker.get(name, np.nan) > 0 for name in data.ticker_names])
    targets = data.target_indices.numpy()[eligible[ids]]
    codes = frame.Ticker.cat.codes.to_numpy()
    if (not frame.Valid.to_numpy()[targets - 1].all()
            or not (codes[targets] == codes[targets - 1]).all()):
        raise ValueError('preceding return crosses a ticker boundary or invalid observation')
    source_dates = frame.Date.to_numpy()[targets]
    preceding = frame.IntradayLogReturn.to_numpy()[targets - 1]
    if not np.isfinite(preceding).all():
        raise ValueError('nonfinite preceding return')

    dates, actuals, predictions = [], [], []
    def emit(batch_dates, batch_actual, batch_predicted):
        dates.append(batch_dates)
        actuals.append(batch_actual)
        predictions.append(batch_predicted)

    result = score_neural(NEURAL[LABEL][0], fit, frame, data, scale_by_ticker,
                          floor, batch_size, emit=emit)
    emitted_dates = np.concatenate(dates)
    actual = np.concatenate(actuals)
    predicted = np.concatenate(predictions)
    if (result['N'] != len(targets)
            or not np.array_equal(emitted_dates, source_dates)
            or not np.allclose(actual, frame.Variance.to_numpy()[targets], rtol=0, atol=0)
            or not np.isfinite(predicted).all() or (predicted <= 0).any()):
        raise ValueError({'scored': result['N'], 'aligned_targets': len(targets),
                          'dates_match': np.array_equal(emitted_dates, source_dates),
                          'actual_match': np.allclose(actual, frame.Variance.to_numpy()[targets],
                                                      rtol=0, atol=0),
                          'predicted_positive': bool(np.isfinite(predicted).all()
                                                     and (predicted > 0).all())})
    means = aggregate_dates(emitted_dates, actual, predicted, preceding)
    if len(means) != 1760 or not np.isfinite(means.to_numpy()).all():
        raise ValueError('expected 1,760 finite global test date means')
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default='D:/DBs/timeseries_analysis/history_coverage.db')
    parser.add_argument('--output', type=Path, default=Path('imgs/data_properties'))
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args()
    frame, _, floor = load_raw_neural(args.database)
    frame = frame.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    fit = artifacts(args.database, frame, floor)[LABEL][1]
    data = TimeSeriesDataset(frame, 80, split='test', valid_column='Valid')
    if len(data) != EXPECTED[80]['test']:
        raise ValueError('unexpected global 80-window test count')
    daily = daily_means(frame, data, scales_from_test(frame, data), fit, floor,
                        args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output / 'global_base_lstm80_daily_means.csv')
    plot_return_curve(daily, args.output / 'global_base_lstm80_return_lowess.png',
                      'Base LSTM-80 Global',
                      'Global equity mean test variance by mean preceding signed return')
    print({'model': LABEL, 'test_dates': len(daily),
           'scored_forecasts': int(daily['count'].sum()),
           'eligible_stocks_per_date': (int(daily['count'].min()), int(daily['count'].max()))})


if __name__ == '__main__':
    main()
