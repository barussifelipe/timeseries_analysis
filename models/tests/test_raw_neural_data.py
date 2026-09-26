"""Small raw-history selection and target-floor check."""

import sqlite3
from contextlib import closing

import pandas as pd
import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from models.raw_neural_data import load_raw_neural
from models.training_blocks import TimeSeriesDataset
from models.variance_neural import make_model, model_data, neural_predict, neural_train, score_neural_records


def test_raw_windows():
    with TemporaryDirectory() as root:
        db = Path(root) / 'raw.db'
        with closing(sqlite3.connect(db)) as conn:
            conn.execute('CREATE TABLE raw_history (Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume REAL)')
            for ticker in ('A', 'B'):
                for i in range(240):
                    date = (pd.Timestamp('2015-01-01') + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                    high = 2. if i != 40 else 1.
                    close = 1. if i != 30 else 1.1
                    low = 1. if i != 40 else 2.
                    volume = None if i == 60 else (-1. if i == 50 else 1.)
                    conn.execute('INSERT INTO raw_history VALUES (?, ?, 1, ?, ?, ?, ?)',
                                 (ticker, date, high, low, close, volume))
                for date, high in (('2016-01-01', 1.), ('2019-01-01', 2.), ('2025-12-31', 2.)):
                    conn.execute('INSERT INTO raw_history VALUES (?, ?, 1, ?, 1, 1, 1)', (ticker, date, high))
            conn.commit()
        frame, history, floor = load_raw_neural(db)
        assert len(history) == 2 and floor > 0
        np.testing.assert_allclose(frame.loc[0, 'RawVariance'], .5 * np.log(2.) ** 2)
        assert frame.loc[(frame.Ticker == 'A') & (frame.Date == '2016-01-01'), 'RawVariance'].item() == 0
        assert frame.loc[(frame.Ticker == 'A') & (frame.Date == '2016-01-01'), 'Variance'].item() == floor
        assert frame.loc[(frame.Ticker == 'A') & (frame.Date == '2015-01-31'), 'IntradayLogReturn'].item() > 0
        assert not frame.loc[(frame.Ticker == 'A') & (frame.Date == '2015-02-10'), 'Valid'].item()
        assert not frame.loc[(frame.Ticker == 'A') & (frame.Date == '2015-02-20'), 'Valid'].item()
        assert not frame.loc[(frame.Ticker == 'A') & (frame.Date == '2015-03-02'), 'Valid'].item()
        assert frame.loc[0, 'IntradayLogReturn'] == 0.
        for width in (20, 80):
            train = TimeSeriesDataset(frame, width, split='train', valid_column='Valid')
            val = TimeSeriesDataset(frame, width, split='val', valid_column='Valid')
            test = TimeSeriesDataset(frame, width, split='test', valid_column='Valid')
            assert all(frame.iloc[i].Ticker == frame.iloc[i + width].Ticker for i in train.valid_indices.tolist())
            assert all(frame.iloc[i].RawVariance > 0 for start in train.valid_indices.tolist() for i in range(start, start + width))
            assert len(val) == 2 and len(test) == 0
            assert np.isfinite(train.data.numpy()).all()
            transformed, column = model_data(frame, 'log-volatility')
            for split, expected in (('train', train), ('val', val), ('test', test)):
                logged = TimeSeriesDataset(transformed, width,
                    feature_columns=(column, 'IntradayLogReturn'), target_column=column,
                    split=split, positive_target=False, valid_column='Valid')
                assert len(logged) == len(expected)
                assert np.isfinite(logged.data.numpy()).all()

        model = make_model('base_lstm_vol', 20, 4, input_size=2)
        fit = {'model_state_dict': model.state_dict(), 'floor': floor,
               'output_convention': 'log_variance',
               'settings': {'model': 'base_lstm_vol', 'output_convention': 'log_variance',
                            'data_transform': 'variance', 'feature_columns': ('Variance', 'IntradayLogReturn'),
                            'window_size': 20, 'hidden_width': 4, 'raw_history': True,
                            'cohort_tickers': tuple(frame.Ticker.unique())}}
        forecasts = neural_predict('base_lstm_vol', fit, frame, 'val')
        assert len(forecasts) == 2 and all(row.actual_variance == floor for row in forecasts)
        assert all(row.predicted_variance >= floor for row in forecasts)
        from models.training_blocks import Forecast
        checked = score_neural_records([Forecast('A', '2019-01-01', 2., 1.),
                                        Forecast('MISSING', '2019-01-01', 2., 1.)],
                                       {'A': np.array([1., 2., 4.])}, raw=True)
        assert checked['observations'] == 1 and checked['excluded_without_training_mase'] == 1
        assert checked['mae'] == 1. and checked['mase'] > 0
        args = SimpleNamespace(database=str(db), estimator='garman-klass', scope='global',
                               ticker=None, limit_tickers=None, limit_rows=None,
                               raw_history=True, intraday_return=True, consecutive_sessions=False,
                               data_transform='log-volatility', window_size=20, hidden_size=128,
                               learning_rate=.001, patience=10, batch_size=128, epochs=30)
        captured = {}
        def inspect(_model, train_data, val_data, _floor, _path, settings, **_kwargs):
            captured.update(settings)
            assert len(train_data) == len(TimeSeriesDataset(frame, args.window_size, split='train', valid_column='Valid'))
            assert len(val_data) == 2
        with patch('models.variance_neural.artifact_path', return_value=Path('unused.pth')), \
             patch('models.variance_neural.wandb_run', return_value=None), \
             patch('models.variance_fit.fit_variance_network', side_effect=inspect):
            neural_train('base_lstm_vol', args)
        assert captured['counts']['val'] == 2
        assert captured['feature_columns'] == ('LogVolatility', 'IntradayLogReturn')
        args.data_transform = 'variance'
        args.training_loss = 'mse'
        with patch('models.variance_neural.artifact_path', return_value=Path('unused.pth')), \
             patch('models.variance_neural.wandb_run', return_value=None), \
             patch('models.variance_fit.fit_variance_network', side_effect=inspect):
            neural_train('silu_lstm', args)
        assert captured['training_loss'] == 'mse'
        assert captured['output_convention'] == 'raw_variance'
        assert captured['feature_columns'] == ('Variance', 'IntradayLogReturn')
        assert captured['counts']['train'] == len(TimeSeriesDataset(frame, 20, split='train', valid_column='Valid'))
        args.window_size = 80
        with patch('models.variance_neural.artifact_path', return_value=Path('unused.pth')), \
             patch('models.variance_neural.wandb_run', return_value=None), \
             patch('models.variance_fit.fit_variance_network', side_effect=inspect):
            neural_train('silu_lstm', args)
            assert captured['counts']['train'] == len(TimeSeriesDataset(frame, 80, split='train', valid_column='Valid'))
            assert captured['training_loss'] == 'mse'
            args.data_transform, args.training_loss = 'log-volatility', 'qlike'
            neural_train('base_lstm_vol', args)
        assert captured['counts']['train'] == len(TimeSeriesDataset(frame, 80, split='train', valid_column='Valid'))
        assert captured['feature_columns'] == ('LogVolatility', 'IntradayLogReturn')
        mse_model = make_model('silu_lstm', 20, 4, input_size=2)
        mse_fit = {'model_state_dict': mse_model.state_dict(), 'floor': floor,
                   'output_convention': 'raw_variance',
                   'settings': {'model': 'silu_lstm', 'output_convention': 'raw_variance',
                                'training_loss': 'mse', 'data_transform': 'variance',
                                'feature_columns': ('Variance', 'IntradayLogReturn'),
                                'window_size': 20, 'hidden_width': 4, 'raw_history': True,
                                'cohort_tickers': tuple(frame.Ticker.unique())}}
        assert all(row.predicted_variance >= floor for row in neural_predict('silu_lstm', mse_fit, frame, 'val'))
        with torch.no_grad():
            mse_model.output_layer.weight.zero_()
            mse_model.output_layer.bias.fill_(-1.)
        mse_fit['model_state_dict'] = mse_model.state_dict()
        assert all(row.predicted_variance == floor for row in neural_predict('silu_lstm', mse_fit, frame, 'val'))
