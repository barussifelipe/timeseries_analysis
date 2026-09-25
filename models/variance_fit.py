"""Shared I/O for independently callable volatility model modules."""

import argparse
import json
import sqlite3
import time
from contextlib import closing

import numpy as np
import torch
from scipy.optimize import minimize

from models.training_blocks import (Forecast, add_common_args, artifact_path,
                             floor_prediction, load_fit, load_variance, prepare, qlike, report_counts,
                             save_fit, variance_metrics, wandb_run)


def neural_log_variance(output, floor, output_convention):
    if output_convention == 'log_variance':
        z = output
    elif output_convention == 'floored_variance':
        z = floor_prediction(output, floor).log()
    else:
        raise ValueError('unknown neural output convention')
    if not torch.isfinite(z).all():
        raise ValueError('neural log-variance forecast must be finite')
    return z


def log_variance_qlike(actual, z, actual_log_volatility=False):
    if not torch.isfinite(actual).all() or (not actual_log_volatility and not (actual > 0).all()):
        raise ValueError('QLIKE requires finite actual log volatility or positive variance')
    log_ratio = (2 * actual if actual_log_volatility else actual.log()) - z
    loss = (torch.expm1(log_ratio) - log_ratio).mean()
    if not torch.isfinite(loss):
        raise ValueError('QLIKE must be finite')
    return loss


def fit_variance_network(model, train_data, val_data, floor, path, settings,
                         epochs=20, batch_size=128, patience=5, run=None):
    """Mean window training loss, epoch variance metrics, one learning-rate retry."""
    from torch.utils.data import DataLoader

    if not len(train_data) or not len(val_data):
        raise ValueError('neural fitting needs train and validation windows')
    if patience < 1:
        raise ValueError('patience must be positive')
    output_convention = settings['output_convention']
    training_loss = settings.get('training_loss', 'qlike')
    if training_loss not in ('qlike', 'mse'):
        raise ValueError('unknown neural training loss')
    log_volatility_target = settings.get('data_transform', 'variance') == 'log-volatility'
    clip_norm = settings.get('clip_norm', 1.)
    log_every = settings.get('log_every_batches', 0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(json.dumps({'training_device': str(device)}), flush=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.get('learning_rate', 1e-3))
    resume = settings.get('resume')
    histories = list(settings.get('resume_history', []))
    best, best_epoch, stale, retried, start_epoch = float('inf'), 0, 0, False, 1
    if resume:
        fit = load_fit(resume)
        for key in ('model', 'window_size', 'hidden_width', 'estimator', 'scope',
                    'run_name', 'batch_size', 'learning_rate', 'output_convention',
                    'clip_norm', 'counts', 'seed', 'data_transform',
                    'consecutive_sessions', 'session_calendar', 'raw_history',
                    'cohort_tickers', 'cohort_rule', 'window_rule', 'target_floor_policy'):
            if fit['settings'].get(key) != settings.get(key):
                raise ValueError(f'resume checkpoint differs in {key}')
        if fit['settings'].get('training_loss', 'qlike') != training_loss:
            raise ValueError('resume checkpoint differs in training_loss')
        if fit['floor'] != floor:
            raise ValueError('resume checkpoint differs in training variance floor')
        if len(histories) != fit['epoch'] or any(row['epoch'] != i for i, row in enumerate(histories, 1)):
            raise ValueError('resume history must end at the best checkpoint epoch')
        model.load_state_dict(fit['model_state_dict'])
        optimizer.load_state_dict(fit['optimizer_state_dict'])
        best, best_epoch, start_epoch = fit[f'val_{training_loss}'], fit['epoch'], fit['epoch'] + 1
        stale, retried = fit.get('stale', 0), fit.get('retried', False)
        if 'torch_rng_state' in fit:
            torch.set_rng_state(fit['torch_rng_state'])
        if device.type == 'cuda' and 'cuda_rng_state' in fit:
            torch.cuda.set_rng_state(fit['cuda_rng_state'])
    if epochs < start_epoch:
        raise ValueError('maximum epochs precedes the resume checkpoint')
    active_model = torch.compile(model) if settings.get('compile') else model
    print(json.dumps({'training_mode': 'compiled' if settings.get('compile') else 'eager',
                      'start_epoch': start_epoch}), flush=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    training = settings['training_history']
    for epoch in range(start_epoch, epochs + 1):
        started = time.monotonic()
        active_model.train()
        clipped_batches = 0
        floor_hits = 0
        train_count = 0
        for batch_index, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = active_model(x)
            if output_convention in ('floored_variance', 'raw_variance'):
                floor_hits += int((output < floor).sum())
                train_count += output.numel()
            if training_loss == 'mse':
                loss = ((output.double() - y.double()) ** 2).mean()
                if not torch.isfinite(loss):
                    raise ValueError('MSE training loss must be finite')
            else:
                z = neural_log_variance(output, floor, output_convention)
                if settings.get('raw_history'):
                    z = z.clamp_min(np.log(floor))
                loss = log_variance_qlike(y, z, log_volatility_target)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   float('inf') if clip_norm is None else clip_norm)
            if not torch.isfinite(norm):
                print(json.dumps({'progress': 'invalid_gradient', 'epoch': epoch,
                                  'batch': batch_index, 'training_loss': training_loss,
                                  'loss': float(loss.detach()),
                                  'max_model_output': float(output.detach().max()),
                                  'max_actual_variance': float(y.detach().max()),
                                  'nonfinite_parameters': [name for name, parameter in model.named_parameters()
                                                           if parameter.grad is not None
                                                           and not torch.isfinite(parameter.grad).all()]}), flush=True)
                raise ValueError('neural gradient must be finite')
            clipped_batches += int(clip_norm is not None and norm > clip_norm)
            optimizer.step()
            if log_every and (batch_index == 1 or batch_index % log_every == 0
                              or batch_index == len(train_loader)):
                print(json.dumps({'progress': 'batch', 'epoch': epoch,
                                  'batch': batch_index, 'batches': len(train_loader),
                                  f'train_{training_loss}_batch': float(loss.detach()),
                                  'gradient_norm': float(norm),
                                  'learning_rate': optimizer.param_groups[0]['lr'],
                                  'elapsed_sec': round(time.monotonic() - started, 1)}), flush=True)
        active_model.eval()
        with torch.no_grad():
            scores = {}
            for split, data, loader in (('train', train_data, DataLoader(train_data, batch_size=batch_size)),
                                        ('val', val_data, val_loader)):
                actual, prediction = [], []
                for batch_index, (x, y) in enumerate(loader, 1):
                    x, y = x.to(device), y.to(device)
                    output = active_model(x)
                    actual_batch = ((2 * y).double().exp() if log_volatility_target else y).flatten()
                    if output_convention == 'raw_variance':
                        prediction_batch = floor_prediction(output.double(), floor).flatten()
                    else:
                        z = neural_log_variance(output, floor, output_convention)
                        if settings.get('raw_history'):
                            z = z.clamp_min(np.log(floor))
                        prediction_batch = z.double().exp().flatten()
                    invalid = (~torch.isfinite(actual_batch) | (actual_batch <= 0)
                               | ~torch.isfinite(prediction_batch) | (prediction_batch <= 0))
                    if invalid.any():
                        position = int(invalid.nonzero()[0, 0])
                        index = len(actual) + position
                        forecast = float(prediction_batch[position])
                        target = float(actual_batch[position])
                        print(json.dumps({'progress': 'invalid_forecast', 'epoch': epoch,
                                          'split': split, 'batch': batch_index, 'index': index,
                                          'ticker': data.ticker_names[int(data.ticker_ids[index])],
                                          'target_date': str(data.target_dates[index].date())
                                          if hasattr(data, 'target_dates') else None,
                                          'model_output': float(output.flatten()[position]),
                                          'variance_forecast': forecast if np.isfinite(forecast) else str(forecast),
                                          'actual_variance': target if np.isfinite(target) else str(target),
                                          'invalid_in_batch': int(invalid.sum())}), flush=True)
                        raise ValueError('QLIKE requires finite positive variance')
                    actual.extend(actual_batch.cpu().tolist())
                    prediction.extend(prediction_batch.cpu().tolist())
                tickers = [data.ticker_names[i] for i in data.ticker_ids.tolist()]
                scores.update({f'{split}/{key}': value for key, value in
                               variance_metrics(actual, prediction, tickers, training).items()})
            score = scores[f'val/{training_loss}']
        if run or log_every:
            row = {'epoch': epoch, **scores,
                     'best_epoch': epoch if score < best else best_epoch,
                     f'best_val_{training_loss}': min(score, best),
                     'train/clipped_batches_pct': 100 * clipped_batches / len(train_loader),
                     **({'train/floor_hits_pct': 100 * floor_hits / train_count}
                        if output_convention in ('floored_variance', 'raw_variance') else {}),
                     'learning_rate': optimizer.param_groups[0]['lr']}
        if log_every:
            print(json.dumps({'progress': 'epoch', **row,
                              'elapsed_sec': round(time.monotonic() - started, 1)}), flush=True)
        if run:
            histories.append(row)
            import wandb
            epochs_run = [item['epoch'] for item in histories]
            run.log({**row, **{f'curves/{metric}': wandb.plot.line_series(
                epochs_run, [[item[f'{split}/{metric}'] for item in histories]
                             for split in ('train', 'val')],
                keys=['train', 'val'], title=metric.upper(), xname='epoch')
                for metric in ('qlike', 'mse', 'rmse', 'mase', 'mae')}})
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            save_fit(path, {'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch, 'val_qlike': scores['val/qlike'],
                            'val_mse': scores['val/mse'], 'floor': floor,
                            'stale': stale, 'retried': retried,
                            'torch_rng_state': torch.get_rng_state(),
                            **({'cuda_rng_state': torch.cuda.get_rng_state()} if device.type == 'cuda' else {}),
                            'output_convention': output_convention,
                            'best_metrics': {key: scores[f'val/{key}'] for key in
                                             ('qlike', 'mse', 'rmse', 'mase', 'mae')},
                            'settings': settings})
        else:
            stale += 1
            if stale >= patience:
                if retried:
                    break
                optimizer.param_groups[0]['lr'] *= .1
                retried, stale = True, 0
    if best == float('inf'):
        raise RuntimeError('no finite validation checkpoint')
    fit = load_fit(path)
    fit['completed_epochs'] = epoch
    fit['settings'] = settings
    save_fit(path, fit)
    print(json.dumps({'progress': 'complete', 'completed_epoch': epoch,
                      'best_epoch': fit['epoch'],
                      f'best_val_{training_loss}': fit[f'val_{training_loss}']}), flush=True)
    return path


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


def rfsv_fit(estimator, source='imgs/roughness_analysis/global/train'):
    import pandas as pd
    from pathlib import Path
    from data.roughness_analysis import QS, scaling_estimates
    source = Path(source).resolve()
    try:
        summary = pd.read_csv(source / 'roughness_summary.csv')
        moments = pd.read_csv(source / 'roughness_moments.csv')
        saved_zeta = pd.read_csv(source / 'roughness_zeta.csv')
    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        raise ValueError('missing RFSV observation-lag training results') from error
    table = f"equity_{estimator.replace('-', '_')}_variance"
    selected = []
    for frame in (summary, moments, saved_zeta):
        required = {'Table', 'Population', 'LagType', 'TrainEnd', 'MaxLag'}
        if not required.issubset(frame.columns):
            raise ValueError('incompatible RFSV training results')
        rows = frame[(frame.Table == table) & (frame.Population == 'global')]
        if rows.empty or not (rows.LagType.eq('observation').all()
                              and rows.TrainEnd.eq('2016-01-01').all()
                              and rows.MaxLag.eq(400).all()):
            raise ValueError('incompatible RFSV training results')
        selected.append(rows)
    summary, moments, saved_zeta = selected
    if (len(summary) != 1 or len(moments) != 400 * len(QS)
            or set(moments.Lag) != set(range(1, 401))
            or not moments.groupby('Lag').size().eq(len(QS)).all()
            or moments.duplicated(['Lag', 'q']).any()
            or set(moments.q) != set(QS)):
        raise ValueError('incomplete RFSV training moments')
    zeta, h, _ = scaling_estimates(moments)
    intercept = zeta.loc[zeta.q == 2, 'Intercept'].item()
    saved_intercept = saved_zeta.loc[saved_zeta.q == 2, 'Intercept'].item()
    if (not np.isfinite(h) or not 0 < h < .5 or not np.isfinite(intercept)
            or not np.isclose(h, summary.H.item())
            or not np.isclose(intercept, saved_intercept)):
        raise ValueError('invalid RFSV training estimates')
    return {'H': float(h), 'nu_squared': float(np.exp(intercept)),
            'lag_type': 'observation', 'max_lag': 400,
            'forecast_window': 20, 'source': str(source)}


def statistical_train(kind, args):
    frame, training, floor = prepare(args)
    minimum = {'ar1': 1, 'har': 20, 'sarima': 1, 'garch': 1, 'rfsv': 20}[kind]
    if args.window_size < minimum:
        raise ValueError(f'{kind} needs at least {minimum} lags')
    if kind == 'ar1' and args.window_size != 1:
        raise ValueError('AR(1) requires exactly one lag')
    if kind == 'rfsv' and args.window_size != 20:
        raise ValueError('RFSV requires exactly 20 observations')
    counts = report_counts(frame, args.window_size)
    diagnostics = {}
    if kind in ('ar1', 'har'):
        params = ols_fit(training, kind, args.window_size)
    elif kind == 'sarima':
        params, diagnostics['converged'] = sarima_fit(training)
    elif kind == 'rfsv':
        params = rfsv_fit(args.estimator)
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
    if kind == 'rfsv' and (not isinstance(params, dict)
            or params.get('lag_type') != 'observation' or params.get('max_lag') != 400
            or params.get('forecast_window') != 20 or window != 20):
        raise ValueError('incompatible RFSV fit')
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
                prediction = RFSV(params['H'], params['nu_squared']).forecast(np.sqrt(history)) ** 2
            else:
                prediction = predictions[i]
            records.append(Forecast(ticker, str(date), float(values[i]),
                                    float(floor_prediction(prediction, fit['floor']))))
    return records


def main(kind):
    parser = add_common_args(argparse.ArgumentParser(description=f'{kind} variance model'),
                             window=20 if kind in ('har', 'rfsv') else 1)
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
