"""Variance target wiring for the four neural architectures."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.training_blocks import (Forecast, TimeSeriesDataset, add_common_args, artifact_path, floor_prediction, load_fit,
                                    load_variance, prepare, qlike, report_counts,
                                    variance_metrics, wandb_run)


def make_model(kind, window, hidden=16):
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
        return SiLULSTM(1, hidden)
    if kind == 'base_lstm_vol':
        return FEBLSTM(1, hidden, 1)
    raise ValueError('unknown neural model')


def model_data(frame, transform):
    if transform == 'variance':
        return frame, 'Variance'
    if transform != 'log-volatility':
        raise ValueError('unknown neural data transform')
    values = frame.Variance.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('log volatility needs finite positive variance')
    frame = frame.copy()
    frame['LogVolatility'] = 0.5 * np.log(values)
    return frame, 'LogVolatility'


def neural_train(kind, args):
    from models.variance_fit import fit_variance_network
    frame, training, floor = prepare(args)
    transform = getattr(args, 'data_transform', 'variance')
    if transform == 'log-volatility' and kind not in ('base_lstm_vol', 'silu_lstm'):
        raise ValueError('log-volatility data are available only for LSTMs')
    frame, column = model_data(frame, transform)
    minimum = {'harnet_20': 20, 'harnet_80': 80}.get(kind)
    if minimum and args.window_size < minimum:
        raise ValueError(f'{kind} needs at least {minimum} lags')
    counts = report_counts(frame, args.window_size)
    train_data = TimeSeriesDataset(frame, args.window_size, feature_columns=(column,),
                                   target_column=column, split='train', positive_target=transform == 'variance')
    val_data = TimeSeriesDataset(frame, args.window_size, feature_columns=(column,),
                                 target_column=column, split='val', positive_target=transform == 'variance')
    torch.manual_seed(42)
    hidden = getattr(args, 'hidden_size', 16)
    learning_rate = getattr(args, 'learning_rate', 1e-3)
    patience = getattr(args, 'patience', 5)
    model = make_model(kind, args.window_size, hidden)
    output_convention = 'floored_variance' if minimum else 'log_variance'
    if minimum:
        from models.variance_fit import ols_fit
        model.initialize_from_har(ols_fit(training, 'har' if minimum == 20 else kind, minimum))
    else:
        output_layer = model.network[-1] if kind == 'mlp' else model.output_layer
        median = float(np.median(np.concatenate(list(training.values()))))
        with torch.no_grad():
            if kind == 'mlp':
                torch.nn.init.xavier_uniform_(output_layer.weight)
            output_layer.weight.mul_(.01)
            output_layer.bias.fill_(np.log(median))
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
                          'data_transform': transform,
                          'clip_norm': None if getattr(args, 'no_grad_clip', False) else 1.,
                          'log_every_batches': getattr(args, 'log_every_batches', 0),
                          'resume': str(path) if getattr(args, 'resume', False) else None,
                          'resume_history': history, 'compile': getattr(args, 'compile', False),
                          'initialization': ('fitted_har' if minimum else 'log_median_xavier_0.01'),
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
    Actual variance is read unchanged from the source frame for exact scoring.
    """
    settings = fit['settings']
    expected = 'floored_variance' if kind.startswith('harnet_') else 'log_variance'
    if fit.get('output_convention') != expected or settings.get('output_convention') != expected or settings.get('model') != kind:
        raise ValueError('incompatible neural checkpoint output convention or model')
    transform = settings.get('data_transform', 'variance')
    frame, column = model_data(frame, transform)
    data = TimeSeriesDataset(frame, settings['window_size'], feature_columns=(column,),
                             target_column=column, split=split, positive_target=transform == 'variance')
    actual_values = frame.sort_values(['Ticker', 'Date']).Variance.to_numpy(dtype=float)
    model = make_model(kind, settings['window_size'], settings.get('hidden_width', 16))
    model.load_state_dict(fit['model_state_dict'])
    model.eval()
    predictions = []
    from models.variance_fit import neural_log_variance
    with torch.no_grad():
        for x, _ in DataLoader(data, batch_size=256):
            output = model(x)
            neural_log_variance(output, fit['floor'], expected)
            variance = (floor_prediction(output, fit['floor'])
                        if expected == 'floored_variance' else output.exp())
            predictions.extend(variance.flatten().tolist())
    return [Forecast(data.ticker_names[data.ticker_ids[i]], str(data.target_dates[i].date()),
                     float(actual_values[int(data.target_indices[i])]),
                     max(predictions[i], fit['floor']) if expected == 'floored_variance' else predictions[i])
            for i in range(len(data))]


def main(kind):
    parser = add_common_args(argparse.ArgumentParser(description=f'{kind} variance model'),
                             window={'harnet_20': 20, 'harnet_80': 80}.get(kind, 30))
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--data-transform', choices=('variance', 'log-volatility'), default='variance')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--no-grad-clip', action='store_true')
    parser.add_argument('--log-every-batches', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-log')
    parser.add_argument('--wandb-id')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    if args.hidden_size < 1 or not 0 < args.learning_rate < float('inf') or args.patience < 1 or args.log_every_batches < 0:
        parser.error('hidden size, learning rate, and patience must be positive; log interval must be nonnegative')
    path = neural_train(kind, args)
    fit = load_fit(path)
    frame, _, _ = prepare(args)
    for split in (('val',) if args.validation_only else ('val', 'test')):
        records = neural_predict(kind, fit, frame, split)
        if records:
            scores = variance_metrics([r[2] for r in records], [r[3] for r in records],
                                      [r[0] for r in records], fit['settings']['training_history'])
            print(json.dumps({'split': split, 'observations': len(records), **scores}))
    if args.crypto_test:
        crypto = load_variance(args.database, args.estimator, asset='crypto',
                               limit_tickers=args.limit_tickers, limit_rows=args.limit_rows)
        records = neural_predict(kind, fit, crypto, 'crypto')
        if records:
            actual, prediction = np.array([(r[2], r[3]) for r in records]).T
            error = actual - prediction
            print(json.dumps({'split': 'crypto', 'observations': len(records),
                              'qlike': qlike(actual, prediction), 'mae': float(np.mean(abs(error))),
                              'mse': float(np.mean(error ** 2)), 'rmse': float(np.sqrt(np.mean(error ** 2)))}))
