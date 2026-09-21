"""Variance target wiring for the four neural architectures."""

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.training_blocks import (Forecast, TimeSeriesDataset, add_common_args, artifact_path, floor_prediction, load_fit,
                                    load_variance, prepare, qlike, report_counts,
                                    variance_metrics, wandb_run)


def make_model(kind, window, hidden=16):
    from models.base_lstm import FEBLSTM
    from models.harnet import HARNet
    from models.mlp import MLP
    from models.silu_lstm import SiLULSTM
    if kind == 'mlp':
        return MLP(window, hidden)
    if kind == 'harnet':
        return HARNet()
    if kind == 'silu_lstm':
        return SiLULSTM(1, hidden)
    if kind == 'base_lstm_vol':
        return FEBLSTM(1, hidden, 1)
    raise ValueError('unknown neural model')


def neural_train(kind, args):
    from models.base_lstm import fit_variance_network
    frame, training, floor = prepare(args)
    if kind == 'harnet' and args.window_size < 20:
        raise ValueError('HARNet needs at least 20 lags')
    counts = report_counts(frame, args.window_size)
    train_data = TimeSeriesDataset(frame, args.window_size, split='train')
    val_data = TimeSeriesDataset(frame, args.window_size, split='val')
    torch.manual_seed(42)
    model = make_model(kind, args.window_size)
    run = wandb_run(args, kind)
    path = artifact_path(kind, args)
    fit_variance_network(model, train_data, val_data, floor, path,
                         {**vars(args), 'model': kind, 'counts': counts,
                          'dates': {'train_end': '2016-01-01', 'val_end': '2019-01-01',
                                    'test_end': '2026-01-01'}, 'training_history': training},
                         epochs=args.epochs, batch_size=args.batch_size, run=run)
    if run:
        run.finish()
    print(path)
    return path


def neural_predict(kind, fit, frame, split='test'):
    """Return Forecast(ticker, target date, actual variance, floored predicted variance).

    Each prediction uses only the preceding ``window_size`` rows of that ticker.
    Actual variance is read unchanged from the source frame for exact scoring.
    """
    settings = fit['settings']
    data = TimeSeriesDataset(frame, settings['window_size'], split=split)
    actual_values = frame.sort_values(['Ticker', 'Date']).Variance.to_numpy(dtype=float)
    model = make_model(kind, settings['window_size'])
    model.load_state_dict(fit['model_state_dict'])
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in DataLoader(data, batch_size=256):
            predictions.extend(floor_prediction(model(x), fit['floor'])
                               .flatten().tolist())
    return [Forecast(data.ticker_names[data.ticker_ids[i]], str(data.target_dates[i].date()),
                     float(actual_values[int(data.target_indices[i])]),
                     max(predictions[i], fit['floor']))
            for i in range(len(data))]


def main(kind):
    parser = add_common_args(argparse.ArgumentParser(description=f'{kind} variance model'),
                             window=20 if kind == 'harnet' else 30)
    args = parser.parse_args()
    path = neural_train(kind, args)
    fit = load_fit(path)
    frame, _, _ = prepare(args)
    for split in ('val', 'test'):
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
