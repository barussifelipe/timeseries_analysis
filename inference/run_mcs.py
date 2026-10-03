"""Score saved five-stock GK forecasts and persist per-stock QLIKE MCS results."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from inference.evaluate_local_global import EXPECTED, path_for, verified_fit
from inference.evaluate_volatility_test import (NEURAL, STATISTICAL, score_neural,
                                               RFSV_WINDOWS, score_statistical)
from inference.run_local_base_lstm import DATABASE, TICKERS
from models.mcs_definition import model_confidence_set, qlike_losses
from models.support_scripts.raw_neural_data import load_raw_neural
from models.training_blocks import TimeSeriesDataset


def collect_forecasts(label, scope, ticker, frame, data, fit, batch_size):
    dates, actuals, predictions = [], [], []

    def emit(d, a, p):
        dates.extend(np.asarray(d).astype('datetime64[D]'))
        actuals.extend(np.asarray(a, dtype=float))
        predictions.extend(np.asarray(p, dtype=float))

    if label in NEURAL:
        score_neural(NEURAL[label][0], fit, frame, data, {ticker: 1.0},
                     fit['floor'], batch_size, emit=emit)
    else:
        score_statistical('rfsv' if label in RFSV_WINDOWS else STATISTICAL[label][0],
                          fit, frame, data, {ticker: 1.0},
                          fit['floor'], emit=emit)
    return (np.asarray(dates, dtype='datetime64[D]'),
            np.asarray(actuals, dtype=float), np.asarray(predictions, dtype=float))


def score_ticker(ticker, database, batch_size, seed, window):
    frame, _, floor = load_raw_neural(database, ticker=ticker)
    frame = frame.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    datasets = {window: TimeSeriesDataset(frame, window, split='test', valid_column='Valid')
                for window in (20, 80)}
    counts = {window: {'train': EXPECTED[window][ticker], 'val': 754, 'test': 1760}
              for window in (20, 80)}
    expected_dates = np.asarray(datasets[window].target_dates).astype('datetime64[D]')
    expected_actual = frame.Variance.to_numpy(dtype=float)[datasets[window].target_indices.numpy()]
    if (len(expected_dates) != 1760 or expected_dates.min() < np.datetime64('2019-01-01')
            or expected_dates.max() > np.datetime64('2025-12-31')
            or np.unique(expected_dates).size != len(expected_dates)):
        raise ValueError(f'{ticker}: mismatched or duplicate test dates')

    names, loss_columns, records, fits = [], [], [], []
    labels = tuple(label for label in (*NEURAL, *STATISTICAL, *RFSV_WINDOWS)
                   if label.endswith(f'-{window}') or label in ('AR(1)', 'GARCH(1,1)', 'SARIMA'))
    for label in labels:
        for scope in ('local', 'global'):
            name = f'{label} {scope.upper()}'
            path, fit = verified_fit(label, scope, ticker, database, floor, counts)
            dates, actual, predicted = collect_forecasts(label, scope, ticker, frame,
                                                          datasets[window], fit, batch_size)
            if (len(dates) != len(expected_dates) or not np.array_equal(dates, expected_dates)
                    or not np.array_equal(actual, expected_actual)):
                raise ValueError(f'{ticker} {name}: forecast keys or actuals differ')
            losses = qlike_losses(actual, predicted)
            names.append(name)
            loss_columns.append(losses)
            fits.append({'candidate': name, 'path': str(path),
                         'artifact_bytes': path.stat().st_size,
                         'artifact_mtime_ns': path.stat().st_mtime_ns,
                         'floor': fit['floor']})
            records.extend({'ticker': ticker, 'date': str(date), 'candidate': name,
                            'actual': float(y), 'predicted': float(p), 'qlike': float(loss)}
                           for date, y, p, loss in zip(dates, actual, predicted, losses))
            print(json.dumps({'ticker': ticker, 'candidate': name, 'n': len(dates),
                              'mean_qlike': float(losses.mean())}), flush=True)

    result = model_confidence_set(np.column_stack(loss_columns), names,
                                  block_length=window, seed=seed)
    return result, records, fits, expected_dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default=DATABASE)
    parser.add_argument('--output', type=Path, default=Path('imgs/eval/mcs'))
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--window', type=int, choices=(20, 80), required=True)
    args = parser.parse_args()
    source = Path(args.database).resolve()
    audit = {(row['ticker'], row['model'], row['scope']): row
             for row in csv.DictReader((Path('imgs/eval/local_eval') / 'per_stock.csv').open(
                 newline='', encoding='utf-8'))}
    seeds = np.random.SeedSequence(42).spawn(len(TICKERS))
    output, all_records = {}, []
    destination = args.output / f'{args.window}_size'
    for ticker, child in zip(sorted(TICKERS), seeds):
        seed = int(child.generate_state(1)[0])
        result, records, fits, dates = score_ticker(ticker, args.database,
                                                     args.batch_size, seed, args.window)
        if len(result['adjusted_p_values']) != 18:
            raise ValueError(f'{ticker}: expected 18 candidates')
        for fit in fits:
            key = (ticker, fit['candidate'].rsplit(' ', 1)[0],
                   fit['candidate'].rsplit(' ', 1)[1])
            if key not in audit:
                raise ValueError(f'{ticker}: missing saved audit row for {fit["candidate"]}')
            measured = np.mean([r['qlike'] for r in records if r['candidate'] == fit['candidate']])
            if not np.isclose(measured, float(audit[key]['QLIKE']), rtol=1e-8, atol=1e-12):
                raise ValueError(f'{ticker} {fit["candidate"]}: QLIKE differs from audit')
        output[ticker] = {'n': len(dates), 'first_date': str(dates[0]),
                          'last_date': str(dates[-1]), 'calendar_gaps': int(np.count_nonzero(
                              np.diff(dates).astype('timedelta64[D]').astype(int) > 1)),
                          'seed': seed, 'fits': fits, **result}
        all_records.extend(records)

    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'forecasts.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    candidates = sorted(output[sorted(TICKERS)[0]]['adjusted_p_values'])
    inclusion = {name: 100 * sum(name in output[ticker]['members'] for ticker in TICKERS) / len(TICKERS)
                 for name in candidates}
    report = {'metadata': {'database': str(source), 'source_bytes': source.stat().st_size,
                           'source_mtime_ns': source.stat().st_mtime_ns,
                           'target': 'next recorded adjusted daily unannualized GK variance',
                           'loss': 'dimensionless QLIKE', 'test_dates': ['2019-01-01', '2025-12-31'],
                           'block_length': args.window, 'repetitions': 10000, 'alpha': 0.10,
                           'candidate_order': candidates,
                           'exclusions': [],
                           'rfsv_history': f'preceding {args.window} valid observations within ticker',
                           'forecast_file': str(destination / 'forecasts.csv')},
              'stocks': output, 'inclusion_pct': inclusion}
    (destination / 'results.json').write_text(json.dumps(report, indent=2) + '\n',
                                             encoding='utf-8')
    shared = sorted(set.intersection(*(set(stock['members']) for stock in output.values())))
    (destination / 'final_sets.md').write_text(
        '# Models in all five final sets\n\n'
        f'Window and bootstrap block: {args.window} observations. '
        'Each stock has 1,760 dated forecasts per candidate.\n\n' +
        ('\n'.join(f'- {name}' for name in shared) if shared else 'No model survives in all five stocks.') +
        '\n', encoding='utf-8')
    print(json.dumps({'results': str(destination / 'results.json'),
                      'forecasts': len(all_records)}), flush=True)


if __name__ == '__main__':
    main()
