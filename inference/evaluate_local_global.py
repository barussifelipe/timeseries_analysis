"""Compare saved local and global GK fits on the same five-stock test targets."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inference.evaluate_volatility_test import (NEURAL, STATISTICAL, ROOT,
                                               scales_from_test, score_neural,
                                               score_rfsv_full, score_statistical)
from inference.plot_global_residuals import NAMES, plot_residuals
from inference.run_local_base_lstm import DATABASE, TICKERS
from models.raw_neural_data import load_raw_neural
from models.training_blocks import TimeSeriesDataset, load_fit


OUTPUT = Path('imgs/eval/local_eval')
EXPECTED = {20: {'NVDA': 4223, 'AAPL': 8408, 'NFLX': 3407, 'GOOG': 2843, 'AMZN': 4669},
            80: {'NVDA': 4103, 'AAPL': 7915, 'NFLX': 3347, 'GOOG': 2783, 'AMZN': 4609}}
CONVENTIONS = {
    'mlp': ('log-volatility', ('LogVolatility', 'IntradayLogReturn'), 'log_variance'),
    'base_lstm_vol': ('log-volatility', ('LogVolatility', 'IntradayLogReturn'), 'log_variance'),
    'silu_lstm': ('variance', ('Variance', 'IntradayLogReturn'), 'raw_variance'),
    'harnet_20': ('variance', ('Variance',), 'floored_variance'),
    'harnet_80': ('variance', ('Variance',), 'floored_variance'),
}


def path_for(label, scope, ticker):
    if label in NEURAL:
        kind, global_run, window = NEURAL[label]
        if scope == 'global':
            return ROOT / kind / 'garman-klass/global' / global_run / 'fit.pth'
        if kind == 'base_lstm_vol':
            run = f'base_lstm_local_{ticker}_raw_gk_logvol_return_w{window}_h128_lr0p001_20e_qlike'
        elif kind == 'mlp':
            run = f'mlp_local_{ticker}_raw_gk_logvol_return_w{window}_h128_lr0p001_20e_qlike'
        elif kind == 'silu_lstm':
            run = f'silu_lstm_local_{ticker}_raw_gk_variance_return_direct_mse_w{window}_h128_lr0p001_20e'
        else:
            run = f'{kind}_local_{ticker}_raw_gk_variance_w{window}_lr0p001_20e_qlike'
        return ROOT / kind / 'garman-klass/local' / ticker / run / 'fit.pth'
    kind, window = (('rfsv', None) if label == 'RFSV-full' else STATISTICAL[label])
    directory = ROOT / kind / 'garman-klass' / scope
    if scope == 'local':
        directory /= ticker
    return directory / ('raw_history_full' if window is None else f'raw_history_w{window}') / 'fit.json'


def verified_fit(label, scope, ticker, database, floor, counts):
    path = path_for(label, scope, ticker)
    fit = load_fit(path) if label in NEURAL else json.loads(path.read_text(encoding='utf-8'))
    source = Path(database).resolve()
    kind, window = (NEURAL[label][0], NEURAL[label][2]) if label in NEURAL else (
        ('rfsv', None) if label == 'RFSV-full' else STATISTICAL[label])
    metadata = fit['settings'] if label in NEURAL else fit
    expected_counts = counts[window] if window else None
    recorded_source = metadata.get('database') if label in NEURAL else metadata.get('source')
    if (metadata.get('model') != kind or metadata.get('scope', 'global') != scope
            or metadata.get('ticker') != (ticker if scope == 'local' else None)
            or Path(recorded_source).resolve() != source
            or (metadata.get('source_bytes') is not None and metadata['source_bytes'] != source.stat().st_size)
            or (metadata.get('source_mtime_ns') is not None and metadata['source_mtime_ns'] != source.stat().st_mtime_ns)
            or (scope == 'local' and metadata.get('cohort_tickers', 1) not in (1, [ticker], (ticker,)))
            or (window and metadata.get('counts') != expected_counts and scope == 'local')
            or (window and (metadata.get('window_size') if label in NEURAL else metadata.get('window')) != window)
            or (scope == 'local' and fit['floor'] != floor)
            or not np.isfinite(fit['floor']) or fit['floor'] <= 0):
        raise ValueError(f'{label} {scope} {ticker}: incompatible source or fit metadata')
    if label in NEURAL:
        transform, features, output = CONVENTIONS[kind]
        if (metadata.get('estimator') != 'garman-klass' or not metadata.get('raw_history')
                or metadata.get('data_transform') != transform
                or tuple(metadata.get('feature_columns', ())) != features
                or metadata.get('output_convention') != output or fit.get('output_convention') != output):
            raise ValueError(f'{label} {scope} {ticker}: incompatible neural input/output convention')
    elif (not fit['diagnostics']['converged'] or fit['source_bytes'] != source.stat().st_size
          or fit['source_mtime_ns'] != source.stat().st_mtime_ns
          or (label == 'RFSV-full' and (fit['parameters']['forecast_window'] != 'full_positive_history'
              or fit['forecast_history_rule'] != 'all preceding valid positive observations within ticker'))):
        raise ValueError(f'{label} {scope} {ticker}: incompatible statistical fit')
    return path, fit


def aggregate(records):
    n = sum(row['N'] for row in records)
    if n <= 0:
        raise ValueError('empty comparison row')
    values = {key: sum(row[key] * row['N'] for row in records) / n
              for key in ('MAE', 'MASE', 'MSE', 'QLIKE', 'floor_hit_pct')}
    return {'N': n, **values, 'RMSE': float(np.sqrt(values['MSE']))}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_table_png(rows, output):
    """Render the matched five-stock CSV rows in the existing metrics-table style."""
    if len(rows) != 28 or sum(row['population'] == 'final-date' for row in rows) != 2:
        raise ValueError('expected 26 full-period and two final-date rows')
    metrics = ('MAE', 'MASE', 'MSE', 'RMSE', 'QLIKE')
    displayed = [[row['model'], f"{int(row['N']):,}",
                  *(f"{float(row[key]):.6g}" for key in metrics),
                  f"{float(row['floor_hit_pct']):.3f}%"] for row in rows]
    fig, ax = plt.subplots(figsize=(17, 14))
    ax.axis('off')
    fig.subplots_adjust(left=.02, right=.98, top=.91, bottom=.11)
    table = ax.table(cellText=displayed,
                     colLabels=['Model', 'N', *metrics, 'Floor hit'],
                     colWidths=[.26, .06, .11, .09, .11, .11, .11, .10],
                     bbox=[0, 0, 1, 1], cellLoc='right')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for index, row in enumerate(rows, start=1):
        table[index, 0].get_text().set_ha('left')
        if row['population'] == 'final-date':
            for col in range(8):
                table[index, col].set_facecolor('#e9edf1')
        elif index % 4 in (3, 0):
            for col in range(8):
                table[index, col].set_facecolor('#f5f7f9')
    full = [index for index, row in enumerate(rows) if row['population'] == 'full-period']
    for col, metric in enumerate(metrics, start=2):
        ranked = sorted(full, key=lambda index: float(rows[index][metric]))
        for index, color in zip(ranked[:2], ('#b00020', '#0057b8')):
            cell = table[index + 1, col]
            cell.get_text().set_color(color)
            cell.get_text().set_weight('bold')
    fig.suptitle('Local vs global one-step adjusted Garman-Klass variance forecasts\n'
                 'NVDA · AAPL · NFLX · GOOG · AMZN | 2019-01-01 to 2025-12-31', fontsize=17, y=.99)
    fig.text(.5, .079, 'Red: lowest full-period loss. Blue: second lowest. All windowed rows use the same 8,800 ticker/date targets.',
             ha='center', fontsize=10)
    fig.text(.5, .053, 'Daily, unannualized variance; MASE uses each ticker’s eligible test-date naive MAE. Floor hit is the share of forecasts raised to the saved floor.',
             ha='center', fontsize=10)
    fig.text(.5, .027, 'Shaded RFSV-full rows use five forecasts on 2025-12-31 only; compare them with each other, not with full-period rows.',
             ha='center', fontsize=10)
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default=DATABASE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--batch-size', type=int, default=2048)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames, datasets, scales, fits = {}, {}, {}, {}
    labels = list(NEURAL) + list(STATISTICAL) + ['RFSV-full']
    for ticker in TICKERS:
        frame, _, floor = load_raw_neural(args.database, ticker=ticker)
        frame = frame.sort_values(['Ticker', 'Date']).reset_index(drop=True)
        frames[ticker] = frame
        counts = {}
        for window in (20, 80):
            data = TimeSeriesDataset(frame, window, split='test', valid_column='Valid')
            if (len(data) != 1760 or data.target_dates.min() < np.datetime64('2019-01-01')
                    or data.target_dates.max() > np.datetime64('2025-12-31')):
                raise ValueError(f'{ticker} window {window}: wrong test targets')
            key = (ticker, window)
            datasets[key] = data
            scales[key] = scales_from_test(frame, data)[ticker]
            if not np.isfinite(scales[key]) or scales[key] <= 0:
                raise ValueError(f'{ticker} window {window}: invalid test MASE scale')
            counts[window] = {'train': EXPECTED[window][ticker], 'val': 754, 'test': 1760}
        for label in labels:
            for scope in ('local', 'global'):
                fits[label, scope, ticker] = verified_fit(label, scope, ticker, args.database, floor, counts)
    table, audit, log = [], [], []
    for label in labels:
        window = NEURAL[label][2] if label in NEURAL else (None if label == 'RFSV-full' else STATISTICAL[label][1])
        per_scope = {}
        for scope in ('local', 'global'):
            records, dates, actuals, predictions = [], [], [], []
            for ticker in TICKERS:
                frame = frames[ticker]
                path, fit = fits[label, scope, ticker]
                def emit(d, a, p):
                    dates.append(d)
                    actuals.append(a)
                    predictions.append(p)
                if label == 'RFSV-full':
                    result, excluded = score_rfsv_full(fit, frame, fit['floor'], emit=emit)
                    if excluded or result['N'] != 1:
                        raise ValueError(f'{label} {scope} {ticker}: final target unavailable')
                elif label in NEURAL:
                    result = score_neural(NEURAL[label][0], fit, frame, datasets[ticker, window],
                                          {ticker: scales[ticker, window]}, fit['floor'], args.batch_size, emit=emit)
                else:
                    result = score_statistical(STATISTICAL[label][0], fit, frame, datasets[ticker, window],
                                               {ticker: scales[ticker, window]}, fit['floor'], emit=emit)
                if result['N'] != (1 if window is None else 1760):
                    raise ValueError(f'{label} {scope} {ticker}: missing forecasts')
                row = {'model': label, 'scope': scope.upper(), 'ticker': ticker,
                       'window': window or 'final-date', 'mase_scale': (scales[ticker, window] if window else
                       abs(float(frame.Variance.iloc[-1]) - float(frame.Variance.iloc[-2]))),
                       'forecast_floor': fit['floor'], **result, 'artifact': str(path)}
                audit.append(row)
                records.append(result)
            per_scope[scope] = (records, np.concatenate(dates).astype('datetime64[D]'))
            combined = aggregate(records)
            row = {'model': f'{label} {scope.upper()}', 'population': 'final-date' if window is None else 'full-period',
                   **combined}
            table.append(row)
            count = plot_residuals(row['model'], dates, actuals, predictions,
                                   args.output / 'residuals' / NAMES[label] / scope)
            if count != combined['N']:
                raise ValueError(f'{label} {scope}: residual count mismatch')
            log.append({'model': row['model'], 'N': count, 'floor_hits_pct': combined['floor_hit_pct']})
        if not np.array_equal(np.sort(per_scope['local'][1]), np.sort(per_scope['global'][1])):
            raise ValueError(f'{label}: LOCAL/GLOBAL target dates differ')
    if len(table) != 28 or len(audit) != 140:
        raise ValueError('incomplete local/global comparison')
    write_csv(args.output / 'metrics.csv', table)
    write_csv(args.output / 'per_stock.csv', audit)
    save_table_png(table, args.output / 'metrics.png')
    source = Path(args.database).resolve()
    header = {'database': str(source), 'source_bytes': source.stat().st_size,
              'source_mtime_ns': source.stat().st_mtime_ns, 'tickers': TICKERS,
              'test_period': ['2019-01-01', '2025-12-31'], 'mase': 'test-date naive MAE per ticker/window'}
    (args.output / 'run.log').write_text('\n'.join(json.dumps(item) for item in [header, *log]) + '\n', encoding='utf-8')
    print(json.dumps({'rows': len(table), 'per_stock_rows': len(audit), 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
