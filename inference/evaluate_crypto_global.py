"""Transfer saved equity global GK fits to retained crypto daily OHLC targets."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inference.evaluate_local_global import aggregate, write_csv
from inference.evaluate_volatility_test import (EXPECTED, NEURAL, RFSV_WINDOWS,
                                                STATISTICAL, ROOT, scales_from_test,
                                                score_neural, score_statistical)
from inference.plot_global_residuals import NAMES, plot_residuals
from models.support_scripts.raw_neural_data import raw_frame
from models.training_blocks import TimeSeriesDataset, load_fit


TICKERS = ('BTC', 'ETH', 'SOL', 'XRP', 'DOGE')
OUTPUT = Path('imgs/eval/crypto_eval')


def crypto_frame(conn, ticker):
    history = pd.read_sql_query('''SELECT symbol AS Ticker, substr(time,1,10) AS Date,
        open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
        FROM crypto_daily_history WHERE symbol = ? AND time < '2026-01-01'
        ORDER BY time''', conn, params=(ticker,))
    retained = pd.read_sql_query('''SELECT Date, Variance FROM crypto_garman_klass_variance
        WHERE Ticker = ? ORDER BY Date''', conn, params=(ticker,))
    if len(retained) != 1760 or history.duplicated('Date').any() or retained.duplicated('Date').any():
        raise ValueError(f'{ticker}: duplicate or incomplete retained dates')
    if history.empty:
        raise ValueError(f'{ticker}: missing daily history')
    calculated = raw_frame(history)
    history = calculated.merge(retained, on='Date', how='left', validate='one_to_one', indicator=True)
    keys = history['_merge'].eq('both').to_numpy()
    if (keys.sum() != 1760 or not history.loc[keys, 'Valid'].all()
            or not (history.loc[keys, 'RawVariance'] > 0).all()
            or not np.allclose(history.loc[keys, 'RawVariance'], history.loc[keys, 'Variance'], rtol=1e-10, atol=1e-14)):
        raise ValueError(f'{ticker}: retained keys disagree with daily OHLC')
    history['Valid'] &= history.RawVariance > 0
    history['Variance'] = history.RawVariance
    history['Date'] = pd.to_datetime(history.Date)
    return history.drop(columns=['_merge']), set(retained.Date)


def selected_dataset(frame, retained, window):
    data = TimeSeriesDataset(frame, window, split='test', valid_column='Valid')
    targets = data.target_indices.numpy()
    dates = frame.Date.iloc[targets].dt.strftime('%Y-%m-%d').to_numpy()
    keep = np.isin(dates, list(retained))
    data.valid_indices = data.valid_indices[keep]
    data.target_indices = data.target_indices[keep]
    data.ticker_ids = data.ticker_ids[keep]
    data.target_dates = data.target_dates[keep]
    if len(data) != 1760 or set(dates[keep]) != retained:
        raise ValueError(f'window {window}: retained targets lack valid prior history')
    return data


def fits(database):
    source = Path(database).resolve()
    stat = source.stat()
    with closing(sqlite3.connect(f'file:{source.as_posix()}?mode=ro', uri=True)) as conn:
        cohort = tuple(row[0] for row in conn.execute('''SELECT Ticker FROM raw_history GROUP BY Ticker
            HAVING SUM(Date < '2016-01-01') > 80 AND SUM(Date = '2025-12-31') > 0 ORDER BY Ticker'''))
    result = {}
    for label, (kind, run, window) in NEURAL.items():
        path = ROOT / kind / 'garman-klass/global' / run / 'fit.pth'
        fit = load_fit(path)
        s = fit['settings']
        if (s.get('model') != kind or s.get('scope', 'global') != 'global'
                or s.get('window_size') != window or s.get('estimator') != 'garman-klass'
                or not s.get('raw_history') or Path(s['database']).resolve() != source
                or (s.get('source_bytes') is not None and s['source_bytes'] != stat.st_size)
                or (s.get('source_mtime_ns') is not None and s['source_mtime_ns'] != stat.st_mtime_ns)
                or tuple(s['cohort_tickers']) != cohort or s.get('counts') != EXPECTED[window]
                or fit.get('output_convention') != s.get('output_convention')
                or not np.isfinite(fit['floor']) or fit['floor'] <= 0):
            raise ValueError(f'{label}: incompatible saved equity fit')
        result[label] = (path, fit, window, kind)
    for label in list(STATISTICAL) + list(RFSV_WINDOWS):
        kind, window = (('rfsv', RFSV_WINDOWS[label]) if label in RFSV_WINDOWS else STATISTICAL[label])
        path = (ROOT / 'rfsv/garman-klass/global/raw_history_full/fit.json' if kind == 'rfsv' else
                ROOT / kind / 'garman-klass/global' / f'raw_history_w{window}' / 'fit.json')
        fit = json.loads(path.read_text(encoding='utf-8'))
        if (fit['model'] != kind or Path(fit['source']).resolve() != source
                or fit['source_bytes'] != stat.st_size or fit['source_mtime_ns'] != stat.st_mtime_ns
                or fit['cohort_tickers'] != len(cohort) or not fit['diagnostics']['converged']
                or not np.isfinite(fit['floor']) or fit['floor'] <= 0
                or (kind == 'rfsv' and (fit['parameters']['forecast_window'] != 'full_positive_history'
                    or fit['forecast_history_rule'] != 'all preceding valid positive observations within ticker'))
                or (kind != 'rfsv' and (fit['window'] != window or fit['counts'] != EXPECTED[window]))):
            raise ValueError(f'{label}: incompatible saved equity fit')
        result[label] = (path, fit, window, kind)
    if len(result) != 15 or len({item[1]['floor'] for item in result.values()}) != 1:
        raise ValueError('incomplete or inconsistent global fits')
    return result


def save_crypto_table(rows, output, target='Garman–Klass'):
    metrics = ('MAE', 'MASE', 'MSE', 'RMSE', 'QLIKE')
    cells = [[row['model'], f"{row['N']:,}",
              *(f"{row[key]:.6g}" for key in metrics),
              f"{row['floor_hit_pct']:.3f}%"] for row in rows]
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=['Model', 'N', *metrics, 'Floor hit'],
                     colWidths=[.24, .09, .105, .105, .105, .105, .105, .145],
                     cellLoc='right', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    for i in range(1, len(rows)+1):
        table[i,0].get_text().set_ha('left')
    ax.set_title('Equity-trained GLOBAL models on BTC, ETH, SOL, XRP, DOGE\n'
                 f'Next recorded daily {target} variance, unannualized', pad=18)
    fig.text(.5, .025, '8,800 identical crypto target keys per row; estimated USD volume selected the assets. '
             'Test-date naive MASE scale per crypto. Separate from equity rankings.', ha='center', fontsize=8)
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default='D:/DBs/timeseries_analysis/history_coverage.db')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--batch-size', type=int, default=2048)
    args = parser.parse_args()
    saved = fits(args.database)
    frames, data, scales, keys = {}, {}, {}, {}
    with closing(sqlite3.connect(f'file:{Path(args.database).resolve().as_posix()}?mode=ro', uri=True)) as conn:
        for ticker in TICKERS:
            frames[ticker], keys[ticker] = crypto_frame(conn, ticker)
            for window in (20, 80):
                data[ticker, window] = selected_dataset(frames[ticker], keys[ticker], window)
                scales[ticker, window] = scales_from_test(frames[ticker], data[ticker, window])[ticker]
                if not np.isfinite(scales[ticker, window]) or scales[ticker, window] <= 0:
                    raise ValueError(f'{ticker} window {window}: invalid MASE scale')
            if not np.array_equal(data[ticker,20].target_dates, data[ticker,80].target_dates):
                raise ValueError(f'{ticker}: window target dates differ')
    args.output.mkdir(parents=True, exist_ok=True)
    table, audit, log = [], [], []
    for label, (path, fit, window, kind) in saved.items():
        records, dates, actuals, predictions = [], [], [], []
        for ticker in TICKERS:
            def emit(d, a, p):
                dates.append(d)
                actuals.append(a)
                predictions.append(p)
            scale = {ticker: scales[ticker, window]}
            result = (score_neural(kind, fit, frames[ticker], data[ticker,window], scale,
                                   fit['floor'], args.batch_size, emit=emit) if label in NEURAL else
                      score_statistical(kind, fit, frames[ticker], data[ticker,window], scale,
                                        fit['floor'], emit=emit))
            if result['N'] != 1760:
                raise ValueError(f'{label} {ticker}: incomplete score')
            records.append(result)
            audit.append({'model': label, 'scope': 'GLOBAL', 'ticker': ticker, 'window': window,
                          'mase_scale': scales[ticker,window], 'forecast_floor': fit['floor'],
                          **result, 'artifact': str(path)})
        combined = aggregate(records)
        table.append({'model': f'{label} GLOBAL', 'population': 'crypto-retained', **combined})
        count = plot_residuals(f'{label} GLOBAL crypto', dates, actuals, predictions,
                               args.output / 'residuals' / NAMES[label] / 'global',
                               variance_label='vendor crypto GK variance')
        if count != combined['N'] or count != 8800:
            raise ValueError(f'{label}: residual count mismatch')
        log.append({'model': label, 'artifact': str(path), 'window': window,
                    'N': count, 'forecast_floor': fit['floor'], 'floor_hit_pct': combined['floor_hit_pct']})
    if len(table) != 15 or len(audit) != 75:
        raise ValueError('incomplete crypto evaluation')
    write_csv(args.output / 'metrics.csv', table)
    write_csv(args.output / 'per_stock.csv', audit)
    save_crypto_table(table, args.output / 'metrics.png')
    source = Path(args.database).resolve()
    header = {'database': str(source), 'source_bytes': source.stat().st_size,
              'source_mtime_ns': source.stat().st_mtime_ns, 'tickers': TICKERS,
              'target_dates': {t: [min(keys[t]), max(keys[t])] for t in TICKERS},
              'target_count_per_ticker': 1760, 'exclusions': 0,
              'target': 'vendor daily unannualized Garman-Klass variance',
              'volume_ranking': 'mean(close * raw volume), estimated USD',
              'mase': 'test-date naive MAE per crypto'}
    (args.output / 'run.log').write_text('\n'.join(json.dumps(x) for x in [header,*log])+'\n', encoding='utf-8')
    print(json.dumps({'rows': len(table), 'per_stock_rows': len(audit), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
