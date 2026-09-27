"""HAR-80 feature and pooled OLS check."""

import numpy as np
import pandas as pd

from models.har_80 import HAR80
from models.raw_statistical_fit import streamed_ols


def test_har_80():
    rng = np.random.default_rng(42)
    values = np.r_[rng.uniform(.001, .01, 120), rng.uniform(.002, .02, 120)]
    frame = pd.DataFrame({'Ticker': ['A'] * 120 + ['B'] * 120,
                          'Date': list(pd.date_range('2015-01-01', periods=120)) * 2,
                          'Variance': values})
    targets = np.r_[np.arange(80, 120), np.arange(200, 240)]
    history = values[:80]
    np.testing.assert_allclose(HAR80.features(history),
                               [1., history[-1], history[-5:].mean(),
                                history[-20:].mean(), history[-40:].mean(), history.mean()])
    x = np.stack([HAR80.features(values[i-80:i]) for i in targets])
    direct = np.linalg.lstsq(x, values[targets], rcond=None)[0]
    streamed, diagnostics = streamed_ols(frame, targets, 'har_80', np.median(values))
    np.testing.assert_allclose(streamed, direct, rtol=1e-9, atol=1e-12)
    assert np.isfinite(diagnostics['normal_matrix_condition'])
    np.testing.assert_allclose(HAR80(streamed).forecast(history), HAR80.features(history) @ direct)
    assert HAR80().fit(values[:120]).shape == (6,)
