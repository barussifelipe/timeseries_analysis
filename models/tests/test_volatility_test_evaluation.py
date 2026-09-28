import csv

import numpy as np
import pandas as pd

from inference.evaluate_volatility_test import (Totals, save_table, scales_from_test,
                                                score_rfsv_full, score_statistical, statistical_predictions)
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
    rfsv = statistical_predictions('rfsv', rfsv_fit, group, positions)
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
        assert 'N' not in saved[0] and 'excluded_no_test_scale' not in saved[0]


def test_full_history_rfsv_one_final_forecast_per_ticker():
    rows = []
    for ticker, values in [('A', [1., 2., 3., 4.]), ('B', [2., 2., 2., 2.])]:
        for i, value in enumerate(values):
            rows.append({'Ticker': ticker, 'Date': pd.Timestamp('2025-12-28') + pd.Timedelta(days=i),
                         'Variance': value, 'RawVariance': value, 'Valid': True})
    frame = pd.DataFrame(rows)
    fit = {'parameters': {'H': .1, 'nu_squared': .2}}
    scored, excluded = score_rfsv_full(fit, frame, 1e-12)
    assert scored['N'] == 1 and excluded == {'B'}
    expected = RFSV(.1, .2).forward([1., 2., 3.])
    assert np.isclose(scored['MAE'], abs(4. - expected))
    assert np.isclose(scored['MASE'], abs(4. - expected))
    changed = frame.copy()
    changed.loc[0, 'Variance'] = changed.loc[0, 'RawVariance'] = 100.
    changed_score, _ = score_rfsv_full(fit, changed, 1e-12)
    assert not np.isclose(scored['MAE'], changed_score['MAE'])
