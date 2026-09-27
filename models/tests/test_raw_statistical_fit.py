"""Small shared-key and pooled-objective check for raw statistical fitting."""

import sqlite3
from contextlib import closing
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from models.raw_statistical_fit import (eligible_data, garch_objective, sarima_objective,
                                        streamed_ols, training_segments)
from models.sarima import SARIMA


def test_raw_statistical_fit():
    with TemporaryDirectory() as directory:
        database = Path(directory) / 'fixture.db'
        with closing(sqlite3.connect(database)) as connection:
            connection.execute('CREATE TABLE raw_history (Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume REAL)')
            for ticker in ('B', 'A'):
                for i in range(100):
                    date = (pd.Timestamp('2015-08-01') + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                    high = 1. if i == 30 else 1.2 + .001 * (i % 13) + .0001 * i * (ticker == 'B')
                    close = 1. if i == 30 else 1. + .01 * (i % 3)
                    volume = -1. if i == 55 else 1.
                    connection.execute('INSERT INTO raw_history VALUES (?, ?, 1, ?, 1, ?, ?)',
                                       (ticker, date, high, close, volume))
                for date in ('2016-01-01', '2019-01-01', '2025-12-31'):
                    connection.execute('INSERT INTO raw_history VALUES (?, ?, 1, 1.2, 1, 1, 1)', (ticker, date))
            connection.commit()
        frame, floor, counts, targets = eligible_data(database, limit_tickers=2)
        assert frame.Ticker.iloc[0] == 'A' and frame.Ticker.iloc[-1] == 'B'
        assert counts == {'train': 78, 'val': 2, 'test': 4}, counts
        assert floor > 0 and frame.loc[0, 'RawVariance'] == .5 * np.log(1.2) ** 2
        assert frame.loc[30, 'Variance'] == floor and not frame.loc[55, 'Valid']
        assert all(frame.iloc[i].Date < pd.Timestamp('2016-01-01') for i in targets)
        assert all(frame.iloc[i].Ticker == frame.iloc[i-20].Ticker for i in targets)
        assert 30 in targets and 55 not in targets and 56 not in targets
        values = frame.Variance.to_numpy(dtype=float)
        scale = float(frame.loc[frame.Date < '2016-01-01', 'Variance'].median())
        for kind in ('ar1', 'har'):
            fitted, _ = streamed_ols(frame, targets, kind, scale)
            x = [[1., values[i-1]] + ([values[i-5:i].mean(), values[i-20:i].mean()] if kind == 'har' else []) for i in targets]
            np.testing.assert_allclose(fitted, np.linalg.lstsq(x, values[targets], rcond=None)[0], rtol=1e-5, atol=1e-9)
        segments = list(training_segments(frame))
        assert len(segments) == 6 and all(len(s[0]) >= 21 for s in segments)
        models = [SARIMA()._model(s[0]) for s in segments]
        parameters = models[0].untransform_params(models[0].start_params)
        np.testing.assert_allclose(sarima_objective(models, parameters),
                                   sum(-model.loglike(parameters, transformed=False) for model in models))
        residuals = [s[1] for s in segments]
        params = [1e-5, .1, .8]
        scale = 1e-4
        np.testing.assert_allclose(garch_objective(residuals, params, scale),
                                   sum(garch_objective([r], params, scale) for r in residuals))
        returns = residuals[0]
        assert np.any(returns != 0)
        causal = returns - np.r_[0., np.cumsum(returns[:-1])] / np.maximum(np.arange(len(returns)), 1)
        state = max(float(np.mean(causal ** 2)), scale * 1e-8)
        manual = 0.
        for error in causal:
            manual += .5 * (np.log(state) + error ** 2 / state)
            state = params[0] + params[1] * error ** 2 + params[2] * state
        np.testing.assert_allclose(garch_objective([causal], params, scale), manual)
