import csv
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inference.evaluate_volatility_test import (Totals, save_table, scales_from_test,
                                                score_statistical, statistical_predictions)
from inference.plot_global_residuals import mean_variances_by_date, plot_residuals
from models.rfsv import RFSV
from models.training_blocks import TimeSeriesDataset, variance_metrics


def test_one_step_statistical_and_output(tmp_path):
    rows = []
    for ticker, multiplier in [('A', 1.), ('B', 2.)]:
        for i in range(24):
            rows.append({'Ticker': ticker, 'Date': pd.Timestamp('2019-01-01') + pd.Timedelta(days=i),
                         'Variance': multiplier * (i + 1), 'RawVariance': multiplier * (i + 1),
                         'IntradayLogReturn': .01 * i, 'Valid': True})
    frame = pd.DataFrame(rows)
    data = TimeSeriesDataset(frame, 20, split='test', valid_column='Valid')
    assert len(data) == 8
    assert data.target_indices.tolist() == [20, 21, 22, 23, 44, 45, 46, 47]
    scales = scales_from_test(frame, data)
    assert scales == {'A': 1., 'B': 2.}
    selected = score_statistical('har', {'parameters': [0, 1, 0, 0]}, frame, data,
                                 scales, 1.)
    assert selected['N'] == 8 and selected['MASE'] == 1.
    emitted = []
    score_statistical('har', {'parameters': [0, 1, 0, 0]}, frame, data, scales, 1.,
                      emit=lambda dates, actual, predicted: emitted.append((dates, actual, predicted)))
    assert np.array_equal(np.concatenate([x[1] - x[2] for x in emitted]), [1.] * 4 + [2.] * 4)
    assert np.array_equal(np.concatenate([x[0] for x in emitted])[:4],
                          frame.Date.iloc[20:24].to_numpy())
    folder = tmp_path / 'residuals'
    log_values = np.linspace(-1., 1., 1000)
    figures = []
    original_close = plt.close
    with patch('inference.plot_global_residuals.plt.close', side_effect=figures.append):
        assert plot_residuals('HAR-20', [pd.date_range('2019-01-01', periods=1000).to_numpy()],
                              [np.exp(log_values)], [np.ones(1000)], folder) == 1000
    raw = np.expm1(log_values)
    raw_skewness = np.mean(((raw - raw.mean()) / raw.std(ddof=0)) ** 3)
    assert f'skewness = {raw_skewness:.4g}' in figures[0]._suptitle.get_text()
    raw_kurtosis = np.mean(((raw - raw.mean()) / raw.std(ddof=0)) ** 4) - 3
    assert f'excess kurtosis = {raw_kurtosis:.4g}' in figures[0]._suptitle.get_text()
    log_residuals = np.log(np.exp(log_values))
    log_skewness = np.mean(((log_residuals - log_residuals.mean()) / log_residuals.std(ddof=0)) ** 3)
    assert f'skewness = {log_skewness:.4g}' in figures[3]._suptitle.get_text()
    log_kurtosis = np.mean(((log_residuals - log_residuals.mean()) / log_residuals.std(ddof=0)) ** 4) - 3
    assert f'excess kurtosis = {log_kurtosis:.4g}' in figures[3]._suptitle.get_text()
    assert np.allclose(figures[1].axes[0].lines[0].get_ydata(), np.expm1(log_values[3:-3]))
    assert np.allclose(figures[4].axes[0].lines[0].get_ydata(), log_values[3:-3])
    for figure in figures:
        original_close(figure)
    assert len(list(folder.glob('*.png'))) == 6
    days, mean_actual, mean_predicted = mean_variances_by_date(
        np.array(['2025-01-02', '2025-01-01', '2025-01-02', '2025-01-01'],
                 dtype='datetime64[D]'),
        np.array([4., 100., 8., 1.]), np.array([2., 90., 4., 2.]))
    assert np.array_equal(days, np.array(['2025-01-01', '2025-01-02'], dtype='datetime64[D]'))
    assert np.array_equal(mean_actual, [50.5, 6.])
    assert np.array_equal(mean_predicted, [46., 3.])
    assert np.log(mean_actual[0]) - np.log(mean_predicted[0]) > 0
    assert np.mean(np.log([100., 1.]) - np.log([90., 2.])) < 0
    mean_figures = []
    with patch('inference.plot_global_residuals.plt.close', side_effect=mean_figures.append):
        plot_residuals('HAR-20', [np.array(['2025-01-02', '2025-01-01',
                                           '2025-01-02', '2025-01-01'], dtype='datetime64[D]')],
                       [np.array([4., 100., 8., 1.])], [np.array([2., 90., 4., 2.])],
                       folder / 'means')
    assert np.allclose(mean_figures[2].axes[0].lines[0].get_ydata(), [4.5, 3.])
    assert np.allclose(mean_figures[5].axes[0].lines[0].get_ydata(),
                       np.log([50.5, 6.]) - np.log([46., 3.]))
    for figure in mean_figures:
        original_close(figure)
    constant_b = frame.copy()
    constant_b.loc[constant_b.Ticker == 'B', ['Variance', 'RawVariance']] = 2.
    scales_with_constant = scales_from_test(constant_b, data)
    assert np.isnan(scales_with_constant['B'])
    assert score_statistical('har', {'parameters': [0, 1, 0, 0]}, constant_b, data,
                             scales_with_constant, 1.)['N'] == 4
    assert score_statistical('har', {'parameters': [0, 1, 0, 0]}, frame, data,
                             scales, 1., {'B'})['N'] == 4
    group = frame.iloc[:24]
    positions = np.array([20, 21])
    har = statistical_predictions('har', {'parameters': [0, 1, 0, 0]}, group, positions)
    assert np.array_equal(har, [20, 21])
    gap = group.copy()
    gap.loc[1, ['Variance', 'RawVariance']] = np.nan
    gap.loc[1, 'Valid'] = False
    assert np.array_equal(statistical_predictions('har', {'parameters': [0, 1, 0, 0]}, gap, positions), [20, 21])
    sarima = statistical_predictions('sarima', {'parameters': [.1, .1, .1, .1, 1.]}, gap, positions)
    assert np.isfinite(sarima).all()
    rfsv_fit = {'parameters': {'H': .1, 'nu_squared': .2}}
    rfsv = statistical_predictions('rfsv', rfsv_fit, group, positions, 20)
    assert np.allclose(rfsv, [RFSV(.1, .2).forecast(np.sqrt(group.Variance.iloc[i-20:i])) ** 2
                              for i in positions])
    actual = np.array([21., 22.])
    totals = Totals()
    totals.add(actual, np.array([-1., 21.]), np.array([2., 2.]), 1.)
    expected = variance_metrics(actual, [1., 21.], ['A', 'A'], {'A': [1., 3.]})
    got = totals.result()
    assert got['N'] == 2 and got['floor_hit_pct'] == 50
    assert all(np.isclose(got[k], expected[k.lower()]) for k in ('MAE', 'MASE', 'MSE', 'RMSE', 'QLIKE'))
    with np.testing.assert_raises(ValueError):
        totals.add([23.], [22.], [np.nan], 1.)
    image = tmp_path / 'metrics.png'
    save_table([{'model': 'HAR-20', **got, 'excluded_no_test_scale': 0, 'artifact': 'fit.json'}], image)
    assert image.is_file() and image.stat().st_size > 0
    with image.with_suffix('.csv').open(newline='', encoding='utf-8') as handle:
        saved = list(csv.DictReader(handle))
        assert saved[0]['model'] == 'HAR-20'
        assert saved[0]['N'] == '2' and 'excluded_no_test_scale' not in saved[0]


def test_window_rfsv_scoring_uses_test_scale():
    rows = []
    for ticker, values in [('A', np.arange(1., 25.)), ('B', np.full(24, 2.))]:
        for i, value in enumerate(values):
            rows.append({'Ticker': ticker, 'Date': pd.Timestamp('2019-01-01') + pd.Timedelta(days=i),
                         'Variance': value, 'RawVariance': value, 'Valid': True})
    frame = pd.DataFrame(rows)
    data = TimeSeriesDataset(frame, 20, split='test', valid_column='Valid')
    scales = scales_from_test(frame, data)
    fit = {'parameters': {'H': .1, 'nu_squared': .2}}
    scored = score_statistical('rfsv', fit, frame, data, scales, 1e-12)
    assert scored['N'] == 4 and np.isnan(scales['B']) and scales['A'] == 1.
    predicted = [RFSV(.1, .2).forward(np.arange(i - 19., i + 1.)) for i in range(20, 24)]
    assert np.isclose(scored['MAE'], np.mean(np.abs(np.arange(21., 25.) - predicted)))
    changed = frame.copy()
    changed.loc[0, 'Variance'] = changed.loc[0, 'RawVariance'] = 100.
    changed_score = score_statistical('rfsv', fit, changed, data, scales, 1e-12)
    assert not np.isclose(scored['MAE'], changed_score['MAE'])
