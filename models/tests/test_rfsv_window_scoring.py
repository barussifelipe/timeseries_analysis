"""RFSV window forecasts use only preceding rows of the selected ticker."""

import numpy as np
import pandas as pd

from inference.evaluate_volatility_test import statistical_predictions
from models.rfsv import RFSV


def test_rfsv_window_scoring():
    fit = {'parameters': {'H': .1, 'nu_squared': .02}}
    model = RFSV(.1, .02)
    for window in (20, 80):
        values = np.linspace(.001, .01, window + 3)
        group = pd.DataFrame({'Variance': values, 'Valid': True})
        positions = np.array([window, window + 1, window + 2])
        predicted = statistical_predictions('rfsv', fit, group, positions, window)
        expected = [model.forward(values[pos - window:pos]) for pos in positions]
        np.testing.assert_allclose(predicted, expected, rtol=1e-12)
        assert np.isfinite(predicted).all() and (predicted > 0).all()
        changed_future = values.copy()
        changed_future[-1] *= 100
        other = statistical_predictions('rfsv', fit,
                                        pd.DataFrame({'Variance': changed_future, 'Valid': True}),
                                        positions[:-1], window)
        np.testing.assert_allclose(other, predicted[:-1], rtol=1e-12)
        # The caller passes ticker groups separately; a new group cannot read prior-ticker rows.
        second = pd.DataFrame({'Variance': values * 3, 'Valid': True})
        actual_second = statistical_predictions('rfsv', fit, second, positions, window)
        np.testing.assert_allclose(actual_second,
                                   [model.forward(second.Variance.iloc[p - window:p]) for p in positions],
                                   rtol=1e-12)


if __name__ == '__main__':
    test_rfsv_window_scoring()
