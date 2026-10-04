"""Score saved equity GK global fits on prior crypto RV and retained RV targets."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from inference.evaluate_crypto_global import (TICKERS, fits, save_crypto_table,
                                              selected_dataset)
from inference.evaluate_local_global import aggregate, write_csv
from inference.evaluate_volatility_test import (NEURAL, scales_from_test,
                                                score_neural, score_statistical)
from inference.plot_global_residuals import NAMES, plot_residuals
from models.support_scripts.raw_neural_data import raw_frame


OUTPUT = Path('imgs/eval/crypto_rv_eval')


def source_range_diagnostics(conn, ticker):
    """Count retained days whose 15-minute closes exceed vendor daily range."""
    row = conn.execute('''SELECT COUNT(*),
        SUM(m.lo < d.low OR m.hi > d.high),
        SUM(m.lo < d.low * .99 OR m.hi > d.high * 1.01),
        SUM(v.Variance > 1), MAX(v.Variance)
        FROM crypto_realized_variance v
        JOIN crypto_daily_history d
          ON d.symbol = v.Ticker AND substr(d.time,1,10) = v.Date
        JOIN (SELECT symbol, substr(time,1,10) AS Date,
                     MIN(close) AS lo, MAX(close) AS hi
              FROM crypto_intraday_history
              WHERE symbol = ? AND time < '2026-01-01'
              GROUP BY symbol, substr(time,1,10)) m
          ON m.symbol = v.Ticker AND m.Date = v.Date
        WHERE v.Ticker = ?''', (ticker, ticker)).fetchone()
    if row[0] != 1760:
        raise ValueError(f'{ticker}: missing 15-minute price range on retained targets')
    return {'intraday_close_outside_daily_range': int(row[1]),
            'outside_daily_range_by_over_1pct': int(row[2]),
            'retained_rv_above_one': int(row[3]), 'maximum_retained_rv': row[4]}


def rv_frame(conn, ticker):
    history = pd.read_sql_query('''SELECT symbol AS Ticker, substr(time,1,10) AS Date,
        open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
        FROM crypto_daily_history WHERE symbol = ? AND time < '2026-01-01'
        ORDER BY time''', conn, params=(ticker,))
    rv = pd.read_sql_query('''SELECT date AS Date, realized_variance AS RV, bars
        FROM crypto_daily_volatility WHERE symbol = ? AND date < '2026-01-01'
        ORDER BY date''', conn, params=(ticker,))
    retained = pd.read_sql_query('''SELECT Date, Variance FROM crypto_realized_variance
        WHERE Ticker = ? ORDER BY Date''', conn, params=(ticker,))
    gk_keys = {row[0] for row in conn.execute(
        'SELECT Date FROM crypto_garman_klass_variance WHERE Ticker = ?', (ticker,))}
    keys = set(retained.Date)
    if (len(retained) != 1760 or len(keys) != 1760 or keys != gk_keys
            or history.empty or history.duplicated('Date').any() or rv.duplicated('Date').any()):
        raise ValueError(f'{ticker}: incomplete, duplicate, or mismatched RV keys')
    frame = raw_frame(history).merge(rv, on='Date', how='left', validate='one_to_one')
    frame = frame.merge(retained, on='Date', how='left', validate='one_to_one', indicator=True)
    target = frame['_merge'].eq('both').to_numpy()
    good = (frame.Valid.to_numpy(dtype=bool) & (frame.bars.to_numpy(dtype=float) == 96)
            & np.isfinite(frame.RV.to_numpy(dtype=float)) & (frame.RV.to_numpy(dtype=float) > 0))
    if (target.sum() != 1760 or not good[target].all()
            or not np.allclose(frame.loc[target, 'RV'], frame.loc[target, 'Variance'],
                               rtol=1e-10, atol=1e-14)):
        raise ValueError(f'{ticker}: retained RV differs from complete 15-minute history')
    omitted = len(frame) - int(good.sum())
    frame = frame.loc[good, ['Ticker', 'Date', 'RV', 'IntradayLogReturn']].copy()
    frame['Valid'] = True
    frame['RawVariance'] = frame.RV
    frame['Variance'] = frame.RV
    frame['Date'] = pd.to_datetime(frame.Date)
    return frame.drop(columns='RV').reset_index(drop=True), keys, omitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default='D:/DBs/timeseries_analysis/history_coverage.db')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--batch-size', type=int, default=2048)
    args = parser.parse_args()
    saved = fits(args.database)
    frames, datasets, scales, keys, coverage = {}, {}, {}, {}, {}
    source = Path(args.database).resolve()
    with closing(sqlite3.connect(f'file:{source.as_posix()}?mode=ro', uri=True)) as conn:
        for ticker in TICKERS:
            frame, keys[ticker], omitted = rv_frame(conn, ticker)
            frames[ticker] = frame
            for window in (20, 80):
                data = selected_dataset(frame, keys[ticker], window)
                datasets[ticker, window] = data
                scales[ticker, window] = scales_from_test(frame, data)[ticker]
                if not np.isfinite(scales[ticker, window]) or scales[ticker, window] <= 0:
                    raise ValueError(f'{ticker} window {window}: invalid RV MASE scale')
            if not np.array_equal(datasets[ticker,20].target_dates,
                                  datasets[ticker,80].target_dates):
                raise ValueError(f'{ticker}: RV window target dates differ')
            dates = frame.Date.to_numpy(dtype='datetime64[D]')
            positions = datasets[ticker,20].target_indices.numpy()
            gap = (dates[positions] - dates[positions-1]) / np.timedelta64(1, 'D')
            coverage[ticker] = {'valid_rv_history_rows': len(frame),
                                'omitted_raw_daily_rows': omitted,
                                'targets_after_calendar_gap': int((gap > 1).sum()),
                                'maximum_target_gap_days': int(gap.max()),
                                'target_first': min(keys[ticker]),
                                'target_last': max(keys[ticker]),
                                **source_range_diagnostics(conn, ticker)}
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
            result = (score_neural(kind, fit, frames[ticker], datasets[ticker,window], scale,
                                   fit['floor'], args.batch_size, emit=emit) if label in NEURAL else
                      score_statistical(kind, fit, frames[ticker], datasets[ticker,window], scale,
                                        fit['floor'], emit=emit))
            if result['N'] != 1760:
                raise ValueError(f'{label} {ticker}: incomplete RV score')
            records.append(result)
            audit.append({'model': label, 'scope': 'GLOBAL', 'ticker': ticker, 'window': window,
                          'mase_scale': scales[ticker,window], 'forecast_floor': fit['floor'],
                          **result, 'artifact': str(path)})
        combined = aggregate(records)
        table.append({'model': f'{label} GLOBAL', 'population': 'crypto-rv-retained', **combined})
        count = plot_residuals(f'{label} GLOBAL crypto RV', dates, actuals, predictions,
                               args.output / 'residuals' / NAMES[label] / 'global',
                               variance_label='15-minute crypto realized variance')
        if count != combined['N'] or count != 8800:
            raise ValueError(f'{label}: RV residual count mismatch')
        log.append({'model': label, 'artifact': str(path), 'window': window,
                    'N': count, 'forecast_floor': fit['floor'], 'floor_hit_pct': combined['floor_hit_pct']})
    if len(table) != 15 or len(audit) != 75:
        raise ValueError('incomplete crypto RV evaluation')
    write_csv(args.output / 'metrics.csv', table)
    write_csv(args.output / 'per_stock.csv', audit)
    save_crypto_table(table, args.output / 'metrics.png', target='15-minute realized')
    header = {'database': str(source), 'source_bytes': source.stat().st_size,
              'source_mtime_ns': source.stat().st_mtime_ns, 'tickers': TICKERS,
              'coverage': coverage, 'target_count_per_ticker': 1760, 'target_exclusions': 0,
              'target': 'daily unannualized sum of 96 squared 15-minute close-to-close log returns',
              'input': 'preceding valid RV observations, daily OHLC intraday return where fitted',
              'fit_origin': 'saved equity Garman-Klass global fits',
              'mase': 'test-date naive MAE per crypto against preceding valid RV observation'}
    (args.output / 'run.log').write_text('\n'.join(json.dumps(x) for x in [header,*log])+'\n', encoding='utf-8')
    print(json.dumps({'rows': len(table), 'per_stock_rows': len(audit), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
