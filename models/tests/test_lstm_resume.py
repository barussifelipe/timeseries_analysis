"""Checkpoint continuation keeps optimizer state, epochs, and W&B identity."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from models.training_blocks import TimeSeriesDataset, load_fit, wandb_run
from models.variance_fit import fit_variance_network


def test_resume():
    frame = pd.DataFrame([('A', day, value) for day, value in zip(
        ('2015-01-01', '2015-01-02', '2015-01-03', '2016-01-01', '2016-01-02'),
        (1., 2., 3., 4., 5.))], columns=('Ticker', 'Date', 'Variance'))
    train = TimeSeriesDataset(frame, 1, split='train')
    val = TimeSeriesDataset(frame, 1, split='val')
    settings = {'model': 'base_lstm_vol', 'window_size': 1, 'hidden_width': 1,
                'learning_rate': 1e-4, 'output_convention': 'log_variance',
                'training_history': {'A': np.array([1., 2., 3.])}}
    class Run:
        def __init__(self):
            self.rows = []
        def log(self, row):
            self.rows.append(row)
    with TemporaryDirectory() as folder:
        path = Path(folder) / 'fit.pth'
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1))
        run = Run()
        fit_variance_network(model, train, val, 1., path, settings,
                             epochs=1, batch_size=2, run=run)
        first = load_fit(path)
        assert first['epoch'] == 1
        assert first['optimizer_state_dict']['state']
        resumed = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1))
        continuation = Run()
        restored = []
        original_load = torch.optim.Adam.load_state_dict
        def record_restore(optimizer, state):
            restored.append(state)
            return original_load(optimizer, state)
        with patch('wandb.plot.line_series', return_value=None), \
             patch.object(torch.optim.Adam, 'load_state_dict', record_restore):
            fit_variance_network(resumed, train, val, 1., path,
                                 {**settings, 'resume': str(path),
                                  'resume_history': run.rows[:1]},
                                 epochs=2, batch_size=2, run=continuation)
        assert continuation.rows[0]['epoch'] == 2
        assert load_fit(path)['completed_epochs'] == 2
        assert restored and next(iter(restored[0]['state'].values()))['step'] >= 1
        class CountingCompiled(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.eval_calls = 0
            def forward(self, x):
                self.eval_calls += int(not torch.is_grad_enabled())
                return self.inner(x)
        compiled_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1))
        wrapper = CountingCompiled(compiled_model)
        with patch.object(torch, 'compile', return_value=wrapper):
            fit_variance_network(compiled_model, train, val, 1., Path(folder) / 'compiled.pth',
                                 {**settings, 'compile': True}, epochs=1, batch_size=2)
        assert wrapper.eval_calls == 2  # one batch for each scored split
    args = type('Args', (), {'no_wandb': False, 'run_name': 'same',
                            'wandb_id': 'kezb18pv'})()
    with patch('wandb.init') as init:
        wandb_run(args, 'base_lstm_vol')
        assert init.call_args.kwargs['id'] == 'kezb18pv'
        assert init.call_args.kwargs['resume'] == 'must'


if __name__ == '__main__':
    test_resume()
