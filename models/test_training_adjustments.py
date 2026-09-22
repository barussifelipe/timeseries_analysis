"""Causal GARCH residuals and fitted HARNet initialization."""

import argparse
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from models.variance_fit import ols_fit, residual_series, statistical_predict
from models.variance_neural import neural_train


def test_garch_causal_residuals_and_forecasts():
    with TemporaryDirectory() as directory:
        database = Path(directory) / 'prices.db'
        with closing(sqlite3.connect(database)) as conn:
            conn.execute('CREATE TABLE raw_history (Ticker TEXT, Date TEXT, Open REAL, Close REAL)')
            conn.execute('CREATE TABLE crypto_daily_history (symbol TEXT, time TEXT, open REAL, close REAL)')
            for ticker, returns in (('A', [1., 2., 3., 4.]), ('B', [4., 2., 1., 0.])):
                for date, value in zip(('2015-12-30', '2015-12-31', '2016-01-01', '2019-01-01'), returns):
                    conn.execute('INSERT INTO raw_history VALUES (?, ?, ?, ?)',
                                 (ticker, date, 1., math.exp(value)))
            for date, value in zip(('2020-01-01', '2020-01-02', '2020-01-03'), (2., 4., 6.)):
                conn.execute('INSERT INTO crypto_daily_history VALUES (?, ?, ?, ?)',
                             ('COIN', date, 1., math.exp(value)))
            conn.commit()
        frame = pd.DataFrame([(ticker, date, .1) for ticker in ('A', 'B')
                              for date in ('2015-12-30', '2015-12-31', '2016-01-01', '2019-01-01')],
                             columns=['Ticker', 'Date', 'Variance'])
        residuals = residual_series(database, frame, 'equity')
        np.testing.assert_allclose(residuals['A'], [1., 1., 1.5, 2.])
        np.testing.assert_allclose(residuals['B'], [4., -2., -2., -7/3])
        crypto = pd.DataFrame([('COIN', date, .1) for date in ('2020-01-01', '2020-01-02', '2020-01-03')],
                              columns=frame.columns)
        np.testing.assert_allclose(residual_series(database, crypto, 'crypto')['COIN'], [2., 2., 3.])
        fit = {'parameters': np.array([.1, .2, .5]), 'floor': 1e-6,
               'settings': {'window_size': 1}}
        first = statistical_predict('garch', fit, frame, 'val', residuals)
        changed = {ticker: values.copy() for ticker, values in residuals.items()}
        changed['A'][2] = 1e4
        unchanged = statistical_predict('garch', fit, frame, 'val', changed)
        assert first[0].predicted_variance == unchanged[0].predicted_variance
        assert statistical_predict('garch', fit, frame, 'test', residuals)[0].predicted_variance != unchanged[0].predicted_variance
        assert len(statistical_predict('garch', fit, crypto, 'crypto', residual_series(database, crypto, 'crypto'))) == 2


def test_harnet_starts_from_fitted_har():
    dates = pd.date_range('2015-09-01', periods=100).strftime('%Y-%m-%d').tolist()
    dates += pd.date_range('2016-01-01', periods=3).strftime('%Y-%m-%d').tolist()
    rows = [(ticker, date, float(base + .01 * i + .03 * math.sin(i / 3)))
            for ticker, base in (('A', 1.), ('B', 2.)) for i, date in enumerate(dates)]
    frame = pd.DataFrame(rows, columns=['Ticker', 'Date', 'Variance'])
    training = {ticker: group[group.Date < '2016-01-01'].Variance.to_numpy()
                for ticker, group in frame.groupby('Ticker')}
    for kind, window, terms in (('harnet_20', 20, (1, 5, 20)),
                                ('harnet_80', 80, (1, 5, 20, 40, 80))):
        fitted = ols_fit(training, 'har' if window == 20 else kind, window)
        args = argparse.Namespace(window_size=window, epochs=1, batch_size=16)
        captured = {}

        with patch('models.variance_neural.prepare', return_value=(frame, training, .01)):
            args.window_size = window - 1
            try:
                neural_train(kind, args)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{kind} accepted short training window')
            args.window_size = window

        def inspect(model, *_args, **_kwargs):
            histories = torch.tensor(np.stack([values[-window:] for values in training.values()]), dtype=torch.float32)
            captured['actual'] = model(histories).detach().numpy().ravel()

        with patch('models.variance_neural.prepare', return_value=(frame, training, .01)), \
             patch('models.variance_neural.report_counts', return_value={'train': 1, 'val': 1, 'test': 0}), \
             patch('models.variance_neural.artifact_path', return_value=Path('unused.pth')), \
             patch('models.variance_neural.wandb_run', return_value=None), \
             patch('models.base_lstm.fit_variance_network', side_effect=inspect):
            neural_train(kind, args)
        expected = [np.dot([1., *(values[-n:].mean() for n in terms)], fitted)
                    for values in training.values()]
        np.testing.assert_allclose(captured['actual'], expected, rtol=1e-5, atol=1e-6)
