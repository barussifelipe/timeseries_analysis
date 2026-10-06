"""Check that the trailing reference excludes the target and stays within a stock."""

import numpy as np
import pandas as pd

from data.plot_data_properties import average_stocks, binned_curve, preceding_reference


def test_preceding_reference():
    actual = preceding_reference(np.array([1., 2., 9., 4.]), window=3)
    assert np.isnan(actual[:3]).all()
    np.testing.assert_allclose(actual[3], 4.)
    assert np.isnan(preceding_reference(np.array([100., 1., 2.]), window=3)).all()


def test_cross_stock_log_values():
    dates = pd.date_range('2019-01-01', periods=1760)
    data = {ticker: pd.DataFrame({'date': dates, 'actual': value,
                                  'predicted': 2 * value, 'preceding_return': value / 100,
                                  'reference': value / 2})
            for ticker, value in zip(('AAPL', 'AMZN', 'GOOG', 'NFLX', 'NVDA'), range(1, 6))}
    mean = average_stocks(data)
    assert len(mean) == 1760
    np.testing.assert_allclose(mean.iloc[0][['actual', 'predicted', 'preceding_return', 'reference']],
                               [3, 6, .03, 1.5])
    np.testing.assert_allclose(np.log(mean.actual / mean.reference), np.log(2))


def test_binned_curve_averages_pairs():
    data = pd.DataFrame({'preceding_return': [-2., -1., 1., 2.],
                         'actual': np.exp([0., 2., 4., 6.]),
                         'predicted': np.exp([1., 3., 5., 7.])})
    curve = binned_curve(data, bin_size=2)
    np.testing.assert_allclose(curve[['return', 'actual', 'predicted']].to_numpy(),
                               [[-1.5, 1., 2.], [1.5, 5., 6.]])
