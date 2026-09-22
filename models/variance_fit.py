"""Shared I/O for independently callable volatility model modules."""

import argparse
import json
import sqlite3
from contextlib import closing

import numpy as np
from scipy.optimize import minimize

from models.training_blocks import (Forecast, add_common_args, artifact_path,
                             floor_prediction, load_fit, load_variance, prepare, qlike, report_counts,
                             save_fit, variance_metrics, wandb_run)


def ols_fit(training, kind, window):
    from models.har import HAR
    x, y = [], []
    lag = window
    for values in training.values():
        for i in range(lag, len(values)):
            if kind == 'ar1':
                features = [1., values[i - 1]]
            elif kind == 'harnet_80':
                features = [1., *(values[i - n:i].mean() for n in (1, 5, 20, 40, 80))]
            else:
                features = HAR.features(values[:i])
            x.append(features)
            y.append(values[i])
    if not x:
        raise ValueError('no fitting lag rows')
    return np.linalg.lstsq(x, y, rcond=None)[0]


def sarima_fit(training):
    from models.sarima import SARIMA
    models = [SARIMA()._model(values) for values in training.values() if len(values) > 10]
    if not models:
        raise ValueError('SARIMA needs complete training series')
    if len(models) == 1:
        result = models[0].fit(disp=False, maxiter=200)
        if not result.mle_retvals.get('converged'):
            result = models[0].fit(start_params=result.params, method='powell',
                                   maxiter=200, disp=False)
        if not result.mle_retvals.get('converged'):
            raise RuntimeError('local SARIMA likelihood did not converge')
        return np.asarray(result.params), bool(result.mle_retvals.get('converged'))
    seed = models[0].fit(disp=False, maxiter=200)
    initial = models[0].untransform_params(seed.params)
    result = minimize(lambda p: -sum(m.loglike(p, transformed=False) for m in models),
                      initial, method='Powell', options={'maxiter': 100})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f'pooled SARIMA fit failed: {result.message}')
    return models[0].transform_params(result.x), True


def residual_series(database, frame, asset):
    """Causal return residuals, restarting the running mean for each ticker."""
    table = 'raw_history' if asset == 'equity' else 'crypto_daily_history'
    symbol = 'Ticker' if asset == 'equity' else 'symbol'
    date = 'Date' if asset == 'equity' else 'substr(time, 1, 10)'
    output = {}
    with closing(sqlite3.connect(database)) as conn:
        for ticker, group in frame.groupby('Ticker'):
            rows = conn.execute(
                f'SELECT {date}, Open, Close FROM {table} WHERE {symbol} = ? ORDER BY {date}',
                (ticker,)).fetchall()
            prices = {d: (o, c) for d, o, c in rows}
            dates = group.sort_values('Date').Date.tolist()
            if any(d not in prices for d in dates):
                raise ValueError(f'{ticker}: missing OHLC for variance dates')
            values = np.array([[*prices[d]] for d in dates], dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError('invalid OHLC')
            returns = np.log(values[:, 1] / values[:, 0])
            past_sum = np.r_[0., np.cumsum(returns[:-1])]
            past_count = np.maximum(np.arange(len(returns)), 1)
            output[ticker] = returns - past_sum / past_count
    return output


def garch_fit(residuals):
    from models.garch import GARCH
    scale = np.mean(np.concatenate([r ** 2 for r in residuals.values()]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('GARCH residual scale must be positive')
    def nll(p):
        omega, alpha, beta = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e100
        total = 0.
        for r in residuals.values():
            variance = max(np.mean(r ** 2), scale * 1e-8)
            for e in r:
                total += 0.5 * (np.log(variance) + e * e / variance)
                variance = omega + alpha * e * e + beta * variance
        return total
    result = minimize(nll, [scale * .1, .1, .8], method='L-BFGS-B',
                      bounds=[(scale * 1e-8, None), (0, 1), (0, 1)],
                      options={'maxiter': 100})
    if not result.success or sum(result.x[1:]) >= 1:
        raise RuntimeError(f'GARCH fit failed: {result.message}')
    GARCH(*result.x)
    return result.x


def rfsv_fit(frame, training):
    import pandas as pd
    from data.roughness_analysis import QS, _series_moments, scaling_estimates
    lags = np.arange(1, min(21, min(map(len, training.values()))), dtype=int)
    sums = np.zeros((len(lags), len(QS)))
    counts = np.zeros(len(lags), dtype=int)
    for ticker, values in training.items():
        dates = frame[(frame.Ticker == ticker) & (frame.Date < '2016-01-01')].Date.to_numpy(dtype='datetime64[D]')
        s, c = _series_moments(dates, .5 * np.log(values), lags, QS)
        sums += s
        counts += c
    rows = [(int(lag), float(q), sums[i, j] / counts[i] if counts[i] else np.nan, int(counts[i]))
            for i, lag in enumerate(lags) for j, q in enumerate(QS)]
    _, h, _ = scaling_estimates(pd.DataFrame(rows, columns=['Lag', 'q', 'Moment', 'Observations']))
    if not np.isfinite(h) or not 0 < h < .5:
        raise ValueError('training-only roughness estimate outside (0, 0.5)')
    return float(h)


def statistical_train(kind, args):
    frame, training, floor = prepare(args)
    minimum = {'ar1': 1, 'har': 20, 'sarima': 1, 'garch': 1, 'rfsv': 1}[kind]
    if args.window_size < minimum:
        raise ValueError(f'{kind} needs at least {minimum} lags')
    if kind == 'ar1' and args.window_size != 1:
        raise ValueError('AR(1) requires exactly one lag')
    counts = report_counts(frame, args.window_size)
    diagnostics = {}
    if kind in ('ar1', 'har'):
        params = ols_fit(training, kind, args.window_size)
    elif kind == 'sarima':
        params, diagnostics['converged'] = sarima_fit(training)
    elif kind == 'rfsv':
        params = rfsv_fit(frame, training)
    else:
        all_residuals = residual_series(args.database, frame, 'equity')
        params = garch_fit({t: r[:len(training[t])] for t, r in all_residuals.items() if t in training})
    fit = {'model': kind, 'parameters': params, 'floor': floor, 'settings': vars(args),
           'dates': {'train_end': '2016-01-01', 'val_end': '2019-01-01', 'test_end': '2026-01-01'},
           'counts': counts, 'fit_observations': (counts['train'] if kind in ('ar1', 'har')
                                                  else sum(map(len, training.values()))),
           'training_history': training, 'diagnostics': diagnostics}
    print(json.dumps({'fit_observations': fit['fit_observations'], 'forecast_windows': counts}))
    if diagnostics:
        print(json.dumps({'fit_diagnostics': diagnostics}))
    path = artifact_path(kind, args)
    save_fit(path, fit)
    run = wandb_run(args, kind)
    if run:
        run.log({'train/count': counts['train'], 'forecast_floor': floor})
        run.finish()
    print(path)
    return path


def statistical_predict(kind, fit, frame, split='test', residuals=None):
    """Return Forecast rows with unchanged actual and floored daily variance."""
    from models.ar1 import AR1
    from models.har import HAR
    from models.sarima import SARIMA
    from models.garch import GARCH
    from models.rfsv import RFSV
    from models.training_blocks import SPLITS
    params = fit['parameters']
    records = []
    window = fit['settings']['window_size']
    start, end = SPLITS[split]
    for ticker, group in frame.groupby('Ticker'):
        group = group.sort_values('Date')
        dates = group.Date.to_numpy(dtype='datetime64[D]')
        values = group.Variance.to_numpy(dtype=float)
        if kind == 'sarima':
            model = SARIMA()._model(values)
            if np.asarray(params).shape != (len(model.param_names),):
                raise ValueError('SARIMA parameter shape changed')
            predictions = model.filter(params).get_prediction(start=0, end=len(values)-1).predicted_mean
        elif kind == 'garch':
            if residuals is None or ticker not in residuals or len(residuals[ticker]) != len(values):
                raise ValueError('GARCH needs aligned residual history')
            garch = GARCH(*params)
            predictions = np.empty(len(values))
            variance = float(garch.omega / (1 - garch.alpha - garch.beta))
            for i, residual in enumerate(residuals[ticker]):
                predictions[i] = variance
                variance = garch.omega + garch.alpha * residual ** 2 + garch.beta * variance
        for i in range(window, len(values)):
            date = dates[i]
            if (start is not None and date < np.datetime64(start)) or (end is not None and date >= np.datetime64(end)):
                continue
            history = values[i-window:i]
            if kind == 'ar1':
                prediction = AR1(*params).forecast(history)
            elif kind == 'har':
                prediction = HAR(params).forecast(history)
            elif kind == 'rfsv':
                prediction = RFSV(params).forecast(np.sqrt(history)) ** 2
            else:
                prediction = predictions[i]
            records.append(Forecast(ticker, str(date), float(values[i]),
                                    float(floor_prediction(prediction, fit['floor']))))
    return records


def main(kind):
    parser = add_common_args(argparse.ArgumentParser(description=f'{kind} variance model'),
                             window=20 if kind == 'har' else 1)
    args = parser.parse_args()
    path = statistical_train(kind, args)
    fit = load_fit(path)
    frame, _, _ = prepare(args)
    residuals = residual_series(args.database, frame, 'equity') if kind == 'garch' else None
    for split in ('val', 'test'):
        records = statistical_predict(kind, fit, frame, split, residuals)
        if records and all(r[0] in fit['training_history'] for r in records):
            scores = variance_metrics([r[2] for r in records], [r[3] for r in records],
                                      [r[0] for r in records], fit['training_history'])
            print(json.dumps({'split': split, 'observations': len(records), **scores}))
    if args.crypto_test:
        crypto = load_variance(args.database, args.estimator, asset='crypto',
                               limit_tickers=args.limit_tickers, limit_rows=args.limit_rows)
        crypto_residuals = residual_series(args.database, crypto, 'crypto') if kind == 'garch' else None
        records = statistical_predict(kind, fit, crypto, 'crypto', crypto_residuals)
        if records:
            actual, prediction = np.array([(r[2], r[3]) for r in records]).T
            error = actual - prediction
            print(json.dumps({'split': 'crypto', 'observations': len(records),
                              'qlike': qlike(actual, prediction), 'mae': float(np.mean(abs(error))),
                              'mse': float(np.mean(error ** 2)), 'rmse': float(np.sqrt(np.mean(error ** 2)))}))
