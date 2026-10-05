"""Small check for GK observation timelines and figure output."""

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from data.plot_gk_data import daily_mean, plot


def test_ticker_date_pooling():
    days = np.array(['2015-01-01', '2015-01-02', '2015-01-01'], dtype='datetime64[D]')
    variance = np.array([1., 9., 3.])
    dates, means = daily_mean(days, variance)
    np.testing.assert_array_equal(dates, np.array(['2015-01-01', '2015-01-02'], dtype='datetime64[D]'))
    np.testing.assert_allclose(means, [2., 9.])
    np.testing.assert_allclose(np.log(means), [np.log(2.), np.log(9.)])
    with TemporaryDirectory() as root:
        plot(days, variance, Path(root), 'Known example')
        assert {p.name for p in Path(root).glob('*.png')} == {
            'timeline_raw.png', 'timeline_log.png', 'histogram_raw.png', 'histogram_log.png'}
