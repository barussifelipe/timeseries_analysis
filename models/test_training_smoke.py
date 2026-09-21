"""Synthetic end-to-end checks; never opens the project database."""

import argparse
import importlib
import os
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from models.variance_fit import residual_series
from models.training_blocks import TimeSeriesDataset, load_fit, load_variance, report_counts
from models.training_blocks import floor_prediction


def test_training_smoke():
    rng = np.random.default_rng(12)
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / 'synthetic.db'
        with closing(sqlite3.connect(database)) as conn:
            for asset in ('equity', 'crypto'):
                for estimator in ('parkinson', 'garman_klass'):
                    conn.execute(f'CREATE TABLE {asset}_{estimator}_variance (Ticker TEXT, Date TEXT, Variance REAL)')
            conn.execute('CREATE TABLE raw_history (Ticker TEXT, Date TEXT, Open REAL, Close REAL)')
            conn.execute('CREATE TABLE crypto_daily_history (symbol TEXT, time TEXT, open REAL, close REAL)')
            dates = list(pd.date_range('2015-09-01', periods=45, freq='D')) + list(pd.date_range('2016-01-01', periods=4, freq='D')) + list(pd.date_range('2019-01-01', periods=4, freq='D'))
            for ticker, base in (('A', .02), ('B', .04)):
                noise = rng.standard_normal(len(dates))
                for i in range(1, len(noise)):
                    noise[i] += .8 * noise[i - 1]
                values = base + .001 * noise
                for date, value in zip(dates, values):
                    d = str(date.date())
                    for estimator in ('parkinson', 'garman_klass'):
                        conn.execute(f'INSERT INTO equity_{estimator}_variance VALUES (?, ?, ?)', (ticker, d, float(value * (1.3 if estimator == 'garman_klass' else 1))))
                    conn.execute('INSERT INTO raw_history VALUES (?, ?, ?, ?)', (ticker, d, 100., 100. * np.exp(rng.normal(0, .1))))
            for i, date in enumerate(pd.date_range('2020-01-01', periods=30, freq='D')):
                d = str(date.date())
                for estimator in ('parkinson', 'garman_klass'):
                    conn.execute(f'INSERT INTO crypto_{estimator}_variance VALUES (?, ?, ?)', ('COIN', d, .03 + i * .0001))
                conn.execute('INSERT INTO crypto_daily_history VALUES (?, ?, ?, ?)', ('COIN', d, 100., 101.))
            conn.commit()
        frame = load_variance(database, 'parkinson')
        limited = load_variance(database, 'parkinson', limit_tickers=1, limit_rows=25)
        assert len(limited) == 33 and len(TimeSeriesDataset(limited, 20, split='val')) == 4
        windows = TimeSeriesDataset(frame, 20, split='val')
        assert len(windows) == 8
        assert report_counts(frame, 20)['val'] == len(windows)
        assert all(str(d.date()).startswith('2016') for d in windows.target_dates)
        assert windows.ticker_names == ['A', 'B']
        for i in range(len(windows)):
            ticker = windows.ticker_names[windows.ticker_ids[i]]
            assert ticker == frame.iloc[int(windows.valid_indices[i]) + 20].Ticker
            assert windows[i][0].shape == (20, 1)
            assert abs(float(windows[i][1]) - frame.iloc[int(windows.valid_indices[i]) + 20].Variance) < 1e-7
            assert windows.target_dates[i] == pd.Timestamp(frame.iloc[int(windows.valid_indices[i]) + 20].Date)
        assert float(floor_prediction(-1, .01)) == .01
        raw = torch.tensor([[-1., 2.]], requires_grad=True)
        mapped = floor_prediction(raw, .01)
        np.testing.assert_allclose(mapped.detach().numpy(), [[.01, 2.]])
        mapped.sum().backward()
        assert raw.grad.tolist() == [[0., 1.]]
        for bad in (0., np.nan, -1.):
            broken = frame.copy()
            broken.loc[0, 'Variance'] = bad
            try:
                TimeSeriesDataset(broken, 1)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid variance accepted')
        original = Path.cwd()
        try:
            os.chdir(directory)
            for kind, estimator, scope in product(
                ('ar1', 'har', 'sarima', 'garch', 'rfsv', 'mlp', 'harnet', 'silu_lstm', 'base_lstm_vol'),
                ('parkinson', 'garman-klass'), ('global', 'local')):
                    module = importlib.import_module('models.base_lstm' if kind == 'base_lstm_vol' else f'models.{kind}')
                    args = argparse.Namespace(database=str(database), estimator=estimator, scope=scope,
                                              ticker='A' if scope == 'local' else None,
                                              run_name='smoke', window_size=20 if kind in ('har', 'harnet') else (1 if kind == 'ar1' else 5),
                                              limit_tickers=None, limit_rows=None, epochs=2, batch_size=16,
                                              no_wandb=True)
                    if kind in ('ar1', 'har', 'sarima', 'garch', 'rfsv'):
                        path = module.train(args)
                        fit = load_fit(path)
                        chosen = load_variance(database, estimator, ticker=args.ticker)
                        residuals = residual_series(database, chosen, 'equity') if kind == 'garch' else None
                        predictions = module.predict(fit, chosen, 'val', residuals)
                        np.testing.assert_array_equal(fit['parameters'], load_fit(path)['parameters'])
                        fixed = np.copy(fit['parameters']) if isinstance(fit['parameters'], np.ndarray) else fit['parameters']
                    else:
                        path = module.train(args)
                        fit = load_fit(path)
                        assert 'output_scale' not in fit
                        chosen = load_variance(database, estimator, ticker=args.ticker)
                        predictions = module.predict(fit, chosen, 'val')
                        assert fit['epoch'] <= args.epochs
                    assert Path(path).exists() and kind in str(path) and estimator in str(path)
                    assert len(predictions) == (8 if scope == 'global' else 4)
                    assert all(p[1].startswith('2016') and p[3] >= fit['floor'] for p in predictions)
                    if kind in ('ar1', 'har', 'sarima', 'garch', 'rfsv'):
                        np.testing.assert_array_equal(fit['parameters'], fixed)
                    if kind == 'sarima':
                        from models.sarima import SARIMA
                        a = chosen[chosen.Ticker == 'A'].sort_values('Date')
                        expected = SARIMA().forecast(a.Variance.to_numpy(dtype=float)[:45], fit['parameters'])
                        assert abs(predictions[0][3] - max(expected, fit['floor'])) < 1e-8
                    crypto = load_variance(database, estimator, asset='crypto')
                    if kind in ('ar1', 'har', 'sarima', 'garch', 'rfsv'):
                        crypto_residuals = residual_series(database, crypto, 'crypto') if kind == 'garch' else None
                        transfer = module.predict(fit, crypto, 'crypto', crypto_residuals)
                    else:
                        transfer = module.predict(fit, crypto, 'crypto')
                    assert transfer and all(row[0] == 'COIN' and row[3] >= fit['floor'] for row in transfer)
                    if kind in ('ar1', 'har', 'sarima', 'garch', 'rfsv'):
                        np.testing.assert_array_equal(fit['parameters'], fixed)
                    assert fit['floor'] == min(min(v) for v in fit.get('training_history', fit['settings'].get('training_history')).values())
        finally:
            os.chdir(original)

        command = [sys.executable, '-m', 'models.har', '--database', str(database),
                   '--estimator', 'parkinson', '--scope', 'global', '--run-name',
                   'cli_smoke', '--no-wandb', '--crypto-test']
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                                env={**os.environ, 'PYTHONPATH': str(original)}, check=True)
        assert '"split": "crypto"' in result.stdout and '"qlike"' in result.stdout

        from models.base_lstm import fit_variance_network
        class Log:
            def __init__(self):
                self.rows = []
            def log(self, row):
                self.rows.append(row)
        log = Log()
        model = torch.nn.Linear(1, 1, bias=False)
        dataset = TensorDataset(torch.zeros(2, 1), torch.ones(2, 1))
        retry_path = Path(directory) / 'retry.pth'
        fit_variance_network(model, dataset, dataset, .1, retry_path, {}, epochs=8,
                             batch_size=2, patience=1, run=log)
        assert load_fit(retry_path)['epoch'] == 1
        assert len(log.rows) == 3 and log.rows[-1]['learning_rate'] < log.rows[0]['learning_rate']


if __name__ == '__main__':
    test_training_smoke()
    print('training smoke passed')
