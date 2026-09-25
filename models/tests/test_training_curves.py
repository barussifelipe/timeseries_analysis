"""Synthetic two-ticker checks for neural training curves."""

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

from models.training_blocks import TimeSeriesDataset, load_fit, variance_metrics
from models.variance_fit import fit_variance_network
from models.variance_neural import make_model, neural_predict


def test_training_curves():
    frame = pd.DataFrame(
        [(ticker, date, value) for ticker, values in
         (('A', [1., 2., 3., 4., 5.]), ('B', [2., 4., 6., 8., 10.]))
         for date, value in zip(('2015-01-01', '2015-01-02', '2015-01-03',
                                 '2016-01-01', '2016-01-02'), values)],
        columns=['Ticker', 'Date', 'Variance'])
    training = {'A': np.array([1., 2., 3.]), 'B': np.array([2., 4., 6.])}
    train = TimeSeriesDataset(frame, 1, split='train')
    val = TimeSeriesDataset(frame, 1, split='val')
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1))
    class Run:
        def __init__(self):
            self.rows = []
        def log(self, row):
            self.rows.append(row)
    run = Run()
    with tempfile.TemporaryDirectory() as directory, patch.object(wandb.plot, 'line_series', side_effect=lambda *a, **k: (a, k)) as plot:
        path = Path(directory) / 'fit.pth'
        fit_variance_network(model, train, val, 1., path,
                             {'output_convention': 'log_variance', 'training_history': training},
                             epochs=2, batch_size=2, run=run)
        assert load_fit(path)['epoch'] in (1, 2)
        assert load_fit(path)['completed_epochs'] == 2
        assert set(load_fit(path)['best_metrics']) == {'qlike', 'mse', 'rmse', 'mase', 'mae'}
        assert len(run.rows) == 2 and plot.call_count == 10
        for epoch, row in enumerate(run.rows[:2], 1):
            assert row['epoch'] == epoch
            best_row = min(run.rows[:epoch], key=lambda item: item['val/qlike'])
            assert row['best_epoch'] == best_row['epoch']
            assert row['best_val_qlike'] == best_row['val/qlike']
            assert set(row) >= {f'{split}/{metric}' for split in ('train', 'val')
                                for metric in ('qlike', 'mse', 'rmse', 'mase', 'mae')}
        # The last epoch's rows must reflect the same final weights on both splits.
        for split, data in (('train', train), ('val', val)):
            actual, predicted = [], []
            with torch.no_grad():
                for x, y in DataLoader(data, batch_size=2):
                    actual.extend(y.flatten().tolist())
                    predicted.extend(model(x.to(next(model.parameters()).device)).exp().cpu().flatten().tolist())
            expected = variance_metrics(actual, predicted,
                                        [data.ticker_names[i] for i in data.ticker_ids.tolist()], training)
            for metric, value in expected.items():
                np.testing.assert_allclose(run.rows[1][f'{split}/{metric}'], value, rtol=1e-6)
        for metric in ('qlike', 'mse', 'rmse', 'mase', 'mae'):
            assert f'curves/{metric}' in run.rows[0]
            chart = run.rows[1][f'curves/{metric}']
            assert chart[0][0] == [1, 2]
            assert chart[0][1] == [[epoch[f'{split}/{metric}'] for epoch in run.rows[:2]]
                                   for split in ('train', 'val')]
            assert chart[1]['keys'] == ['train', 'val']
        with patch.object(wandb.plot, 'line_series', side_effect=AssertionError('W&B used')):
            fit_variance_network(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1)),
                                 train, val, 1., Path(directory) / 'offline.pth',
                                 {'output_convention': 'log_variance', 'training_history': training},
                                 epochs=1, batch_size=2, run=None)
        model32 = make_model('base_lstm_vol', 1, 32)
        with torch.no_grad():
            model32.output_layer.weight.mul_(.01)
            model32.output_layer.bias.fill_(np.log(3.))
        settings = {'model': 'base_lstm_vol', 'window_size': 1, 'hidden_width': 32,
                    'learning_rate': 1e-4, 'output_convention': 'log_variance',
                    'training_history': training}
        wide_path = Path(directory) / 'wide.pth'
        fit_variance_network(model32, train, val, 1., wide_path, settings,
                             epochs=1, batch_size=2)
        fit = load_fit(wide_path)
        assert fit['optimizer_state_dict']['param_groups'][0]['lr'] == 1e-4
        assert fit['settings']['hidden_width'] == 32
        assert len(neural_predict('base_lstm_vol', fit, frame, 'val')) == len(val)
        unclipped = Run()
        terminal = io.StringIO()
        with patch.object(torch.nn.utils, 'clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as norm_check, \
             redirect_stdout(terminal):
            fit_variance_network(torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1)),
                                 train, val, 1., Path(directory) / 'unclipped.pth',
                                 {'output_convention': 'log_variance', 'training_history': training,
                                  'clip_norm': None, 'log_every_batches': 1},
                                 epochs=1, batch_size=2, run=unclipped)
        assert all(call.args[1] == float('inf') for call in norm_check.call_args_list)
        assert unclipped.rows[0]['train/clipped_batches_pct'] == 0
        assert load_fit(Path(directory) / 'unclipped.pth')['settings']['clip_norm'] is None
        progress = [json.loads(line) for line in terminal.getvalue().splitlines()]
        batches = [row for row in progress if row.get('progress') == 'batch']
        assert [row['batch'] for row in batches] == list(range(1, len(train) // 2 + 1))
        assert all(row['gradient_norm'] >= 0 and np.isfinite(row['train_qlike_batch'])
                   for row in batches)
        epoch = next(row for row in progress if row.get('progress') == 'epoch')
        assert epoch['val/qlike'] == unclipped.rows[0]['val/qlike']
        assert epoch['train/clipped_batches_pct'] == 0

        extreme = TensorDataset(torch.zeros(1, 1, 1), torch.zeros(1, 1))
        extreme.ticker_names, extreme.ticker_ids = ['A'], torch.tensor([0])
        extreme_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1))
        with torch.no_grad():
            extreme_model[-1].weight.zero_()
            extreme_model[-1].bias.fill_(100.)
        extreme_run = Run()
        fit_variance_network(extreme_model, extreme, extreme, 1.,
                             Path(directory) / 'extreme.pth',
                             {'output_convention': 'log_variance', 'data_transform': 'log-volatility',
                              'training_history': {'A': [1., 2.]}},
                             epochs=1, run=extreme_run)
        assert np.isfinite(extreme_run.rows[0]['val/mse'])
        assert 98 < extreme_run.rows[0]['val/qlike'] < 100

        overflow = TensorDataset(torch.zeros(1, 1, 1), torch.zeros(1, 1))
        overflow.ticker_names, overflow.ticker_ids = ['A'], torch.tensor([0])
        overflow.target_dates = pd.DatetimeIndex(['2015-01-03'])
        with torch.no_grad():
            extreme_model[-1].bias.fill_(710.)
        terminal = io.StringIO()
        with redirect_stdout(terminal):
            try:
                fit_variance_network(extreme_model, overflow, overflow, 1.,
                                     Path(directory) / 'overflow.pth',
                                     {'output_convention': 'log_variance',
                                      'data_transform': 'log-volatility',
                                      'training_history': {'A': [1., 2.]}}, epochs=1)
            except ValueError as error:
                assert str(error) == 'QLIKE requires finite positive variance'
            else:
                raise AssertionError('overflowed forecast was accepted')
        offender = next(json.loads(line) for line in terminal.getvalue().splitlines()
                        if '"progress": "invalid_forecast"' in line)
        assert (offender['epoch'], offender['split'], offender['ticker'], offender['target_date']) == (
            1, 'train', 'A', '2015-01-03')
        assert offender['model_output'] > 709
        assert offender['variance_forecast'] == 'inf'
        assert offender['actual_variance'] == 1.

        mse_data = TensorDataset(torch.zeros(2, 1, 1), torch.tensor([[1.], [3.]]))
        mse_data.ticker_names, mse_data.ticker_ids = ['A'], torch.tensor([0, 0])
        mse_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1))
        with torch.no_grad():
            mse_model[-1].weight.zero_()
            mse_model[-1].bias.fill_(2.)
        mse_run = Run()
        mse_path = Path(directory) / 'mse.pth'
        terminal = io.StringIO()
        with redirect_stdout(terminal):
            fit_variance_network(mse_model, mse_data, mse_data, 1., mse_path,
                                 {'output_convention': 'raw_variance', 'training_loss': 'mse',
                                  'learning_rate': 0., 'log_every_batches': 1,
                                  'training_history': {'A': [1., 2., 3.]}},
                                 epochs=1, batch_size=2, run=mse_run)
        batch = next(json.loads(line) for line in terminal.getvalue().splitlines()
                     if '"progress": "batch"' in line)
        np.testing.assert_allclose(batch['train_mse_batch'], 1., rtol=1e-6)
        np.testing.assert_allclose(mse_run.rows[0]['train/mse'], 1., rtol=1e-6)
        np.testing.assert_allclose(mse_run.rows[0]['best_val_mse'], 1., rtol=1e-6)
        assert 'best_val_qlike' not in mse_run.rows[0]
        np.testing.assert_allclose(load_fit(mse_path)['val_mse'], 1., rtol=1e-6)
        with torch.no_grad():
            mse_model[-1].bias.fill_(173.)
        fit_variance_network(mse_model, mse_data, mse_data, 1.,
                             Path(directory) / 'mse_extreme.pth',
                             {'output_convention': 'raw_variance', 'training_loss': 'mse',
                              'learning_rate': 0., 'training_history': {'A': [1., 2., 3.]}},
                             epochs=1, batch_size=2)
        with torch.no_grad():
            mse_model[-1].bias.fill_(-1.)
        negative_run = Run()
        terminal = io.StringIO()
        with redirect_stdout(terminal):
            fit_variance_network(mse_model, mse_data, mse_data, 1.,
                                 Path(directory) / 'mse_negative.pth',
                                 {'output_convention': 'raw_variance', 'training_loss': 'mse',
                                  'learning_rate': 0., 'log_every_batches': 1,
                                  'training_history': {'A': [1., 2., 3.]}},
                                 epochs=1, batch_size=2, run=negative_run)
        batch = next(json.loads(line) for line in terminal.getvalue().splitlines()
                     if '"progress": "batch"' in line)
        assert batch['train_mse_batch'] == 10.
        assert negative_run.rows[0]['val/mse'] == 2.
        assert negative_run.rows[0]['train/floor_hits_pct'] == 100.


if __name__ == '__main__':
    test_training_curves()
