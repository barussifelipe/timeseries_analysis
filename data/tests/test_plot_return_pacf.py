"""Check that residual-correlation PACF recovers a known AR(1) pattern."""

import numpy as np

from data.plot_return_pacf import partial_correlations


def test_partial_correlations():
    rng = np.random.default_rng(0)
    innovations = rng.normal(size=5000)
    values = np.empty(len(innovations))
    values[0] = innovations[0]
    for i in range(1, len(values)):
        values[i] = 0.6 * values[i - 1] + innovations[i]
    result = partial_correlations(values, 3)
    assert abs(result[0] - 0.6) < 0.05
    assert max(abs(result[1:])) < 0.05
