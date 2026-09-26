"""Variance target wiring for the four neural architectures."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.training_blocks import (Forecast, TimeSeriesDataset, add_common_args, artifact_path, floor_prediction, load_fit,
                                    load_variance, prepare, qlike, report_counts,
                                    variance_metrics, wandb_run)


def make_model(kind, window, hidden=16, input_size=1):
    from models.base_lstm import FEBLSTM
    from models.harnet_20 import HARNet as HARNet20
    from models.harnet_80 import HARNet as HARNet80
    from models.mlp import MLP
    from models.silu_lstm import SiLULSTM
    if kind == 'mlp':
        return MLP(window, hidden)
    if kind == 'harnet_20':
        return HARNet20()
    if kind == 'harnet_80':
        return HARNet80()
    if kind == 'silu_lstm':
        return SiLULSTM(input_size, hidden)
    if kind == 'base_lstm_vol':
        return FEBLSTM(input_size, hidden, 1)
    raise ValueError('unknown neural model')


def raw_har_fit(data, window):
    """Fit HAR initialization on the exact eligible training windows."""
    lengths = (1, 5, 20) if window == 20 else (1, 5, 20, 40, 80)
    values = data.data[:, 0].numpy().astype(float)
    targets = data.targets[:, 0].numpy().astype(float)
    starts = data.valid_indices.numpy()
    gram = np.zeros((len(lengths) + 1, len(lengths) + 1))
    cross = np.zeros(len(lengths) + 1)
    for block in np.array_split(starts, max(1, (len(starts) + 4095) // 4096)):
        if not len(block):
            continue
        x = np.column_stack([np.ones(len(block)), *(np.stack([values[i + window - n:i + window] for i in block]).mean(axis=1) for n in lengths)])
        gram += x.T @ x
        cross += x.T @ targets[block + window]
    return np.linalg.lstsq(gram, cross, rcond=None)[0]


def model_data(frame, transform):
    if transform == 'variance':
        return frame, 'Variance'
    if transform != 'log-volatility':
        raise ValueError('unknown neural data transform')
    values = frame.Variance.to_numpy(dtype=float)
    valid = frame.Valid.to_numpy(dtype=bool) if 'Valid' in frame else np.ones(len(frame), dtype=bool)
    if not np.isfinite(values[valid]).all() or (values[valid] <= 0).any():
        raise ValueError('log volatility needs finite positive variance')
    frame = frame.copy()
    logged = np.full(len(values), np.nan)
    logged[valid] = 0.5 * np.log(values[valid])
    frame['LogVolatility'] = logged
    return frame, 'LogVolatility'


def neural_train(kind, args):
    from models.variance_fit import fit_variance_network
    raw = getattr(args, 'raw_history', False)
    if raw and kind not in ('base_lstm_vol', 'silu_lstm', 'harnet_20', 'harnet_80'):
        raise ValueError('raw-history path is defined for the four requested models')
    use_return = kind in ('base_lstm_vol', 'silu_lstm') and (raw or getattr(args, 'intraday_return', False))
    if raw:
        if args.scope == 'local' and not args.ticker or args.scope == 'global' and args.ticker:
            raise ValueError('local raw-history runs need a ticker; global runs cannot select one')
        from models.raw_neural_data import load_raw_neural
        frame, training, floor = load_raw_neural(args.database, args.ticker, args.limit_tickers, args.limit_rows)
    else:
        frame, training, floor = prepare(args, include_intraday_log_return=use_return)
    transform = getattr(args, 'data_transform', 'variance')
    training_loss = getattr(args, 'training_loss', 'qlike')
    if training_loss == 'mse' and (kind not in ('base_lstm_vol', 'silu_lstm') or transform != 'variance' or not raw):
        raise ValueError('MSE training is defined here for raw-variance LSTMs')
    if transform == 'log-volatility' and kind not in ('base_lstm_vol', 'silu_lstm'):
        raise ValueError('log-volatility data are available only for LSTMs')
    frame, column = model_data(frame, transform)
    features = (column, 'IntradayLogReturn') if use_return else (column,)
    minimum = {'harnet_20': 20, 'harnet_80': 80}.get(kind)
    if minimum and args.window_size < minimum:
        raise ValueError(f'{kind} needs at least {minimum} lags')
    if raw and (args.estimator != 'garman-klass' or getattr(args, 'consecutive_sessions', False)):
        raise ValueError('raw-history runs require Garman-Klass and next recorded observation')
    allowed_windows = {'harnet_20': (20,), 'harnet_80': (80,)}.get(kind, (20, 80))
    if raw and args.window_size not in allowed_windows:
        raise ValueError('raw-history window must be 20 or 80 for LSTMs, 20 for HARNet-20, or 80 for HARNet-80')
    consecutive = getattr(args, 'consecutive_sessions', False)
    calendar = tuple(sorted(str(date)[:10] for date in frame.Date.unique())) if consecutive else None
    valid_column = 'Valid' if raw else None
    counts = (None if raw else report_counts(frame, args.window_size, consecutive_sessions=consecutive,
                           session_calendar=calendar))
    train_data = TimeSeriesDataset(frame, args.window_size, feature_columns=features,
                                   target_column=column, split='train', positive_target=transform == 'variance',
                                   consecutive_sessions=consecutive, session_calendar=calendar, valid_column=valid_column)
    val_data = TimeSeriesDataset(frame, args.window_size, feature_columns=features,
                                 target_column=column, split='val', positive_target=transform == 'variance',
                                 consecutive_sessions=consecutive, session_calendar=calendar, valid_column=valid_column)
    if raw:
        counts = {split: len(TimeSeriesDataset(frame, args.window_size, feature_columns=features,
                  target_column=column, split=split, positive_target=transform == 'variance',
                  valid_column='Valid'))
                  for split in ('train', 'val', 'test')}
        print(json.dumps({'forecast_windows': counts}))
    torch.manual_seed(42)
    hidden = getattr(args, 'hidden_size', 16)
    learning_rate = getattr(args, 'learning_rate', 1e-3)
    patience = getattr(args, 'patience', 5)
    model = make_model(kind, args.window_size, hidden, input_size=len(features))
    output_convention = ('floored_variance' if minimum else
                         'raw_variance' if training_loss == 'mse' else 'log_variance')
    if minimum:
        from models.variance_fit import ols_fit
        model.initialize_from_har(ols_fit(training, 'har' if minimum == 20 else kind, minimum)
                                  if not raw else raw_har_fit(train_data, minimum))
    else:
        output_layer = model.network[-1] if kind == 'mlp' else model.output_layer
        median = float(np.median(np.concatenate(list(training.values()))))
        with torch.no_grad():
            if kind == 'mlp':
                torch.nn.init.xavier_uniform_(output_layer.weight)
            output_layer.weight.mul_(.01)
            output_layer.bias.fill_(median if training_loss == 'mse' else np.log(median))
    path = artifact_path(kind, args)
    history = []
    if getattr(args, 'resume', False):
        if not getattr(args, 'resume_log', None):
            raise ValueError('--resume-log is required with --resume')
        history = [json.loads(line) for line in Path(args.resume_log).read_text(encoding='utf-8').splitlines()
                   if line.startswith('{"progress": "epoch"')]
        best_epoch = load_fit(path)['epoch']
        superseded = [row['epoch'] for row in history if row['epoch'] > best_epoch]
        history = [row for row in history if row['epoch'] <= best_epoch]
        print(json.dumps({'resume_best_epoch': best_epoch,
                          'superseded_epochs': superseded}), flush=True)
    run = wandb_run(args, kind)
    if run and getattr(args, 'resume', False):
        run.log({'restart/from_best_epoch': best_epoch,
                 'restart/superseded_epochs': len(superseded),
                 'restart/patience': patience,
                 'restart/compiled': int(getattr(args, 'compile', False))})
    fit_variance_network(model, train_data, val_data, floor, path,
                         {**vars(args), 'model': kind, 'counts': counts,
                          'output_convention': output_convention, 'seed': 42,
                          'hidden_width': hidden, 'learning_rate': learning_rate, 'patience': patience,
                          'data_transform': transform, 'training_loss': training_loss,
                          'feature_columns': features,
                          'consecutive_sessions': consecutive, 'session_calendar': calendar,
                          'raw_history': raw, 'cohort_tickers': tuple(frame.Ticker.unique()) if raw else None, 'cohort_rule': '>80 pre-2016 raw rows and 2025-12-31 row' if raw else None,
                          'window_rule': 'positive valid inputs; valid target; next recorded observation' if raw else None,
                          'target_floor_policy': 'zero target plus minimum positive pre-2016 variance' if raw else None,
                          'clip_norm': None if getattr(args, 'no_grad_clip', False) else 1.,
                          'log_every_batches': getattr(args, 'log_every_batches', 0),
                          'resume': str(path) if getattr(args, 'resume', False) else None,
                          'resume_history': history, 'compile': getattr(args, 'compile', False),
                          'initialization': ('fitted_har' if minimum else
                                             'raw_median_xavier_0.01' if training_loss == 'mse' else
                                             'log_median_xavier_0.01'),
                          'dates': {'train_end': '2016-01-01', 'val_end': '2019-01-01',
                                    'test_end': '2026-01-01'}, 'training_history': training},
                         epochs=args.epochs, batch_size=args.batch_size, patience=patience, run=run)
    if run:
        print(json.dumps({'wandb_url': run.url}), flush=True)
        run.finish()
    print(path)
    return path


def neural_predict(kind, fit, frame, split='test'):
    """Return Forecast(ticker, target date, actual variance, predicted variance).

    Each prediction uses only the preceding ``window_size`` rows of that ticker.
    In raw mode actual variance is the adjusted target; RawVariance retains the original.
    """
    settings = fit['settings']
    expected = ('floored_variance' if kind.startswith('harnet_') else
                'raw_variance' if settings.get('training_loss') == 'mse' else 'log_variance')
    if fit.get('output_convention') != expected or settings.get('output_convention') != expected or settings.get('model') != kind:
        raise ValueError('incompatible neural checkpoint output convention or model')
    transform = settings.get('data_transform', 'variance')
    frame, column = model_data(frame, transform)
    features = tuple(settings.get('feature_columns', (column,)))
    if settings.get('raw_history'):
        if ('Valid' not in frame or 'RawVariance' not in frame
                or tuple(frame.Ticker.unique()) != tuple(settings['cohort_tickers'])):
            raise ValueError('raw-history checkpoint requires its original cohort and raw-history frame')
        if frame.loc[frame.Valid & (frame.Date < '2016-01-01') & (frame.RawVariance > 0),
                     'RawVariance'].min() != fit['floor']:
            raise ValueError('raw-history checkpoint floor differs from source data')
    data = TimeSeriesDataset(frame, settings['window_size'], feature_columns=features,
                             target_column=column, split=split, positive_target=transform == 'variance',
                             consecutive_sessions=settings.get('consecutive_sessions', False),
                             session_calendar=settings.get('session_calendar') if split != 'crypto' else None,
                             valid_column='Valid' if settings.get('raw_history') else None)
    actual_values = frame.sort_values(['Ticker', 'Date']).Variance.to_numpy(dtype=float)
    model = make_model(kind, settings['window_size'], settings.get('hidden_width', 16),
                       input_size=len(features))
    model.load_state_dict(fit['model_state_dict'])
    model.eval()
    predictions = []
    from models.variance_fit import neural_log_variance
    with torch.no_grad():
        for x, _ in DataLoader(data, batch_size=256):
            output = model(x)
            variance = (floor_prediction(output.double(), fit['floor']) if expected == 'raw_variance' else
                        floor_prediction(output, fit['floor']) if expected == 'floored_variance' else
                        output.double().exp())
            if expected != 'raw_variance':
                neural_log_variance(output, fit['floor'], expected)
            if not torch.isfinite(variance).all() or not (variance > 0).all():
                raise ValueError('neural forecast must be finite positive variance')
            predictions.extend(variance.flatten().tolist())
    return [Forecast(data.ticker_names[data.ticker_ids[i]], str(data.target_dates[i].date()),
                     float(actual_values[int(data.target_indices[i])]),
                     max(predictions[i], fit['floor']) if expected == 'floored_variance' or settings.get('raw_history') else predictions[i])
            for i in range(len(data))]


def score_neural_records(records, training, raw=False):
    scoring = [r for r in records if r.ticker in training] if raw else records
    if not scoring:
        raise ValueError('no forecasts have a training-history MASE scale')
    scores = variance_metrics([r.actual_variance for r in scoring],
                              [r.predicted_variance for r in scoring],
                              [r.ticker for r in scoring], training)
    return {'observations': len(scoring),
            'excluded_without_training_mase': len(records) - len(scoring), **scores}


def _main(kind):
    parser = add_common_args(argparse.ArgumentParser(description=f'{kind} variance model'),
                             window={'harnet_20': 20, 'harnet_80': 80}.get(kind, 30))
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--data-transform', choices=('variance', 'log-volatility'), default='variance')
    parser.add_argument('--training-loss', choices=('qlike', 'mse'), default='qlike')
    parser.add_argument('--raw-history', action='store_true', help='use eligible raw adjusted-OHLC cohort and valid windows')
    parser.add_argument('--intraday-return', action='store_true', help='add ln(adjusted Close/Open) to LSTM inputs')
    parser.add_argument('--consecutive-sessions', action='store_true',
                        help='require every input and target date to be consecutive observed market sessions')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--no-grad-clip', action='store_true')
    parser.add_argument('--log-every-batches', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-log')
    parser.add_argument('--wandb-id')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    if args.raw_history and args.crypto_test:
        parser.error('--raw-history is limited to equity inference')
    if args.intraday_return and kind not in ('base_lstm_vol', 'silu_lstm'):
        parser.error('--intraday-return applies only to LSTMs')
    if args.hidden_size < 1 or not 0 < args.learning_rate < float('inf') or args.patience < 1 or args.log_every_batches < 0:
        parser.error('hidden size, learning rate, and patience must be positive; log interval must be nonnegative')
    path = neural_train(kind, args)
    fit = load_fit(path)
    if args.raw_history:
        from models.raw_neural_data import load_raw_neural
        frame, _, _ = load_raw_neural(args.database, args.ticker, args.limit_tickers, args.limit_rows)
    else:
        frame, _, _ = prepare(args, include_intraday_log_return=args.intraday_return)
    for split in (('val',) if args.validation_only else ('val', 'test')):
        records = neural_predict(kind, fit, frame, split)
        if records:
            scores = score_neural_records(records, fit['settings']['training_history'],
                                           fit['settings'].get('raw_history', False))
            print(json.dumps({'split': split, **scores}))
    if args.crypto_test:
        crypto = load_variance(args.database, args.estimator, asset='crypto',
                               limit_tickers=args.limit_tickers, limit_rows=args.limit_rows,
                               include_intraday_log_return=args.intraday_return)
        records = neural_predict(kind, fit, crypto, 'crypto')
        if records:
            actual, prediction = np.array([(r[2], r[3]) for r in records]).T
            error = actual - prediction
            print(json.dumps({'split': 'crypto', 'observations': len(records),
                              'qlike': qlike(actual, prediction), 'mae': float(np.mean(abs(error))),
                              'mse': float(np.mean(error ** 2)), 'rmse': float(np.sqrt(np.mean(error ** 2)))}))


def main(kind):
    started = time.monotonic()
    status = 'failed'
    try:
        _main(kind)
        status = 'complete'
    except KeyboardInterrupt:
        status = 'interrupted'
        raise
    finally:
        print(json.dumps({'progress': 'run_end', 'status': status,
                          'elapsed_sec': round(time.monotonic() - started, 1)}), flush=True)
