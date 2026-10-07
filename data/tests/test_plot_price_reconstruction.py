"""Small known examples for price reconstruction inputs and simulation."""

import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from data.plot_price_reconstruction import (MLP_GLOBAL, PLOTS, daily, daily_mean_logs, daily_gross_returns,
                                            log_daily_means, main, preceding_ar1_mean, simulate)


def test_price_reconstruction():
    dates = pd.date_range('2018-01-01', periods=255)
    frame = pd.DataFrame({'Ticker': pd.Categorical(['A'] * 255 + ['B'] * 255),
                          'Date': list(dates) * 2,
                          'Valid': [True] * 254 + [False] + [True] * 255,
                          'IntradayLogReturn': [0.01] * 254 + [np.nan] + [0.02] * 255})
    drift = preceding_ar1_mean(frame, np.array([251, 252, 253, 254, 255, 507]))
    assert np.isnan(drift[[0, 3, 4]]).all()
    np.testing.assert_allclose(drift[[1, 2, 5]], [0.01, 0.01, 0.02])
    frame.loc[100, 'Valid'] = False
    frame.loc[100, 'IntradayLogReturn'] = np.nan
    assert np.isnan(preceding_ar1_mean(frame, np.array([252]))[0])
    np.testing.assert_allclose(preceding_ar1_mean(frame, np.array([253]))[0], 0.01)
    returns = np.random.default_rng(0).normal(0, 0.02, 255)
    ar_frame = pd.DataFrame({'Ticker': pd.Categorical(['A'] * 255),
                             'Valid': True, 'IntradayLogReturn': returns})
    coefficients = np.linalg.lstsq(np.column_stack((np.ones(251), returns[:251])),
                                    returns[1:252], rcond=None)[0]
    expected_ar = coefficients[0] + coefficients[1] * returns[251]
    np.testing.assert_allclose(preceding_ar1_mean(ar_frame, np.array([252])), [expected_ar])
    ar_frame.loc[252, 'IntradayLogReturn'] = 99.
    np.testing.assert_allclose(preceding_ar1_mean(ar_frame, np.array([252])), [expected_ar])
    expected = 100 * np.exp(0.01 + 0.2 * 0.5)
    np.testing.assert_allclose(simulate(np.array([100.]), np.array([0.04]),
                                        np.array([0.01]), np.array([0.5])), [expected])
    for opened, variance in ((0., 0.04), (100., 0.), (np.inf, 0.04)):
        try:
            simulate(np.array([opened]), np.array([variance]), np.array([0.01]), np.array([0.5]))
        except ValueError:
            pass
        else:
            raise AssertionError('invalid adjusted Open or variance accepted')
    values = pd.DataFrame({'date': [dates[252]] * 2, 'open': [100., 200.],
                           'actual_close': [110., 90.],
                           'simulated_close': [expected, 100.]})
    result = daily(values)
    assert result['count'].iloc[0] == 2
    assert result.actual_close.iloc[0] == 100
    np.testing.assert_allclose(result.simulated_close.iloc[0], (expected + 100) / 2)
    logged = log_daily_means(result)
    np.testing.assert_allclose(logged.log_actual_mean_close.iloc[0], np.log(100))
    np.testing.assert_allclose(logged.log_simulated_mean_close.iloc[0],
                               np.log((expected + 100) / 2))
    assert logged['count'].equals(result['count'])
    mean_logged = daily_mean_logs(values)
    np.testing.assert_allclose(mean_logged.mean_log_actual_close.iloc[0],
                               (np.log(110) + np.log(90)) / 2)
    np.testing.assert_allclose(mean_logged.mean_log_simulated_close.iloc[0],
                               (np.log(expected) + np.log(100)) / 2)
    assert mean_logged['count'].equals(result['count'])
    gross = daily_gross_returns(values)
    np.testing.assert_allclose(gross.actual_gross_return.iloc[0], (1.1 + 0.45) / 2)
    np.testing.assert_allclose(gross.simulated_gross_return.iloc[0], (expected / 100 + 0.5) / 2)
    assert gross['count'].equals(result['count'])


def test_show_saved_plots():
    from matplotlib import pyplot as plt

    plt.switch_backend('Agg')
    with TemporaryDirectory() as temp:
        for name, _ in PLOTS:
            pd.DataFrame({'date': ['2025-01-01'], 'actual_close': [100.],
                          'simulated_close': [101.], 'count': [2]}).to_csv(Path(temp) / f'{name}.csv', index=False)
            pd.DataFrame({'date': ['2025-01-01'], 'log_actual_mean_close': [np.log(100.)],
                          'log_simulated_mean_close': [np.log(101.)], 'count': [2]}).to_csv(
                              Path(temp) / f'{name}_log.csv', index=False)
            pd.DataFrame({'date': ['2025-01-01'], 'actual_gross_return': [1.01],
                          'simulated_gross_return': [1.02], 'count': [2]}).to_csv(
                              Path(temp) / f'{name}_gross_return.csv', index=False)
        pd.DataFrame({'date': ['2025-01-01'], 'actual_gross_return': [1.01],
                      'simulated_gross_return': [1.02], 'count': [2]}).to_csv(
                          Path(temp) / f'{MLP_GLOBAL[0]}.csv', index=False)
        with (patch('sys.argv', ['plot_price_reconstruction', '--output', temp, '--show']),
              patch.object(plt, 'switch_backend'), patch.object(plt, 'show') as show):
            main()
            assert len(plt.get_fignums()) == 7
            for figure_number in (3, 6):
                ax = plt.figure(figure_number).axes[0]
                np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.01])
                np.testing.assert_allclose(ax.lines[1].get_ydata(), [0.02])
                np.testing.assert_allclose(ax.lines[2].get_ydata(), [0., 0.])
            show.assert_called_once()
    plt.close('all')
