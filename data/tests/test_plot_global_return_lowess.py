"""Check that global date means keep matched returns and variance units."""

import numpy as np
import pandas as pd

from data.plot_global_return_lowess import aggregate_dates


def test_aggregate_dates():
    days = pd.to_datetime(['2019-01-02', '2019-01-02', '2019-01-03'])
    result = aggregate_dates(days, [1., 3., 2.], [2., 4., 8.], [-.02, .04, .01])
    np.testing.assert_allclose(result.iloc[0].to_numpy(), [2., 3., .01, 2.])
    np.testing.assert_allclose(result.iloc[1].to_numpy(), [2., 8., .01, 1.])
    np.testing.assert_allclose(np.log(result.actual.iloc[0]), np.log(2.))
