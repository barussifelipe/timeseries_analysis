"""Log-volatility inputs and targets keep variance-unit scoring."""

import argparse
import sqlite3
from contextlib import closing
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.training_blocks import TimeSeriesDataset, load_variance
from models.variance_fit import log_variance_qlike
from models.variance_neural import make_model, model_data, neural_predict, neural_train


def test_log_volatility_inputs():
    rows = [(ticker, date, float(np.exp(2 * (offset + i))))
            for ticker, offset in (('A', 0), ('B', 4))
            for i, date in enumerate(('2015-01-01', '2015-01-02', '2015-01-03', '2016-01-01'))]
    frame = pd.DataFrame(rows[::-1], columns=('Ticker', 'Date', 'Variance'))
    transformed, feature = model_data(frame, 'log-volatility')
    assert feature == 'LogVolatility'
    np.testing.assert_allclose(transformed.LogVolatility, [7, 6, 5, 4, 3, 2, 1, 0])
    train = TimeSeriesDataset(transformed, 2, feature_columns=(feature,), target_column=feature,
                              split='train', positive_target=False)
    val = TimeSeriesDataset(transformed, 2, feature_columns=(feature,), target_column=feature,
                            split='val', positive_target=False)
    assert len(train) == len(val) == 2
    np.testing.assert_allclose(train[0][0].flatten(), [0, 1])
    np.testing.assert_allclose(train[1][0].flatten(), [4, 5])
    np.testing.assert_allclose(val[0][0].flatten(), [1, 2])
    np.testing.assert_allclose(val[1][0].flatten(), [5, 6])
    np.testing.assert_allclose([val[i][1].item() for i in range(2)], [3, 7])
    y = torch.tensor([[-2.], [3.]])
    z = torch.tensor([[-3.], [5.]])
    torch.testing.assert_close(log_variance_qlike(y, z, True),
                               log_variance_qlike((2 * y).exp(), z))
    model = make_model('base_lstm_vol', 2, 2)
    fit = {'model_state_dict': model.state_dict(), 'floor': 1.,
           'output_convention': 'log_variance',
           'settings': {'model': 'base_lstm_vol', 'window_size': 2, 'hidden_width': 2,
                        'output_convention': 'log_variance', 'data_transform': 'log-volatility'}}
    forecasts = neural_predict('base_lstm_vol', fit, frame, 'val')
    with torch.no_grad():
        expected = [model(x).exp().item() for x, _ in DataLoader(val, batch_size=1)]
    np.testing.assert_allclose([row.predicted_variance for row in forecasts], expected)
    np.testing.assert_allclose([row.actual_variance for row in forecasts], [np.exp(6), np.exp(14)], rtol=1e-6)
    try:
        model_data(frame.assign(Variance=0.), 'log-volatility')
    except ValueError:
        pass
    else:
        raise AssertionError('nonpositive variance accepted before logarithm')


if __name__ == '__main__':
    test_log_volatility_inputs()


def test_intraday_log_return_feature(tmp_path):
    database = tmp_path / 'tiny.db'
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute('CREATE TABLE equity_garman_klass_variance (Ticker TEXT, Date TEXT, Variance REAL)')
        conn.execute('CREATE TABLE raw_history (Ticker TEXT, Date TEXT, Open REAL, Close REAL)')
        for ticker, offset in (('A', 0), ('B', 4)):
            for i, date in enumerate(('2015-01-01', '2015-01-02', '2015-01-03', '2016-01-01')):
                conn.execute('INSERT INTO equity_garman_klass_variance VALUES (?, ?, ?)',
                             (ticker, date, float(np.exp(2 * (offset + i)))))
                conn.execute('INSERT INTO raw_history VALUES (?, ?, ?, ?)',
                             (ticker, date, 100., 100. * np.exp(.1 * (offset + i))))
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute('CREATE TABLE crypto_garman_klass_variance (Ticker TEXT, Date TEXT, Variance REAL)')
        conn.execute('CREATE TABLE crypto_daily_history (symbol TEXT, time TEXT, open REAL, close REAL)')
        conn.execute("INSERT INTO crypto_garman_klass_variance VALUES ('COIN', '2020-01-01', 0.01)")
        conn.execute("INSERT INTO crypto_daily_history VALUES ('COIN', '2020-01-01 00:00:00', 100, 110)")
    crypto = load_variance(database, 'garman-klass', asset='crypto', include_intraday_log_return=True)
    np.testing.assert_allclose(crypto.IntradayLogReturn, [np.log(1.1)])
    frame = load_variance(database, 'garman-klass', include_intraday_log_return=True)
    np.testing.assert_allclose(frame.IntradayLogReturn, [.0, .1, .2, .3, .4, .5, .6, .7])
    transformed, target = model_data(frame, 'log-volatility')
    features = (target, 'IntradayLogReturn')
    val = TimeSeriesDataset(transformed, 2, feature_columns=features,
                            target_column=target, split='val', positive_target=False)
    np.testing.assert_allclose(val[0][0], [[1, .1], [2, .2]])
    np.testing.assert_allclose(val[1][0], [[5, .5], [6, .6]])
    for kind in ('base_lstm_vol', 'silu_lstm'):
        model = make_model(kind, 2, 2, input_size=2)
        fit = {'model_state_dict': model.state_dict(), 'floor': 1.,
               'output_convention': 'log_variance',
               'settings': {'model': kind, 'window_size': 2, 'hidden_width': 2,
                            'output_convention': 'log_variance', 'data_transform': 'log-volatility',
                            'feature_columns': features}}
        forecasts = neural_predict(kind, fit, frame, 'val')
        with torch.no_grad():
            expected = [model(x).exp().item() for x, _ in DataLoader(val, batch_size=1)]
        np.testing.assert_allclose([row.predicted_variance for row in forecasts], expected)
        args = argparse.Namespace(database=str(database), estimator='garman-klass', scope='global',
                                  ticker=None, limit_tickers=None, limit_rows=None, window_size=2,
                                  hidden_size=2, learning_rate=.001, patience=2, epochs=1,
                                  batch_size=2, no_wandb=True, data_transform='log-volatility',
                                  run_name='tiny', intraday_return=True)
        seen = {}
        def capture(model, train, validation, floor, path, settings, **kwargs):
            seen.update(model=model, train=train, validation=validation, settings=settings)
        with patch('models.variance_fit.fit_variance_network', capture), \
             patch('models.variance_neural.artifact_path', return_value=tmp_path / 'fit.pth'):
            neural_train(kind, args)
        assert seen['train'].input_size == seen['validation'].input_size == 2
        assert seen['model'].input_size == 2
        assert tuple(seen['settings']['feature_columns']) == features
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute("DELETE FROM raw_history WHERE Ticker = 'A' AND Date = '2015-01-01'")
    try:
        load_variance(database, 'garman-klass', include_intraday_log_return=True)
    except ValueError as error:
        assert 'matched finite positive' in str(error)
    else:
        raise AssertionError('missing Open/Close accepted')
