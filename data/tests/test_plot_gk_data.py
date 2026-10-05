"""Small check for GK observation timelines and figure output."""

from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
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
        figures = []
        with patch('data.plot_gk_data.plt.close', side_effect=figures.append):
            plot(days, variance, Path(root), 'Known example')
        for figure, values in zip(figures[2:], (variance, np.log(variance))):
            standardized = (values - values.mean()) / values.std(ddof=0)
            title = figure._suptitle.get_text()
            assert f'skewness = {np.mean(standardized ** 3):.4g}' in title
            assert f'excess kurtosis = {np.mean(standardized ** 4) - 3:.4g}' in title
        for figure in figures:
            plt.close(figure)
        full_figures = []
        with patch('data.plot_gk_data.plt.close', side_effect=full_figures.append):
            plot(days, variance, Path(root), 'Known example', full_only=True)
        assert all(len(figure.axes) == 1 for figure in full_figures[2:])
        for figure in full_figures:
            plt.close(figure)
        assert {p.name for p in Path(root).glob('*.png')} == {
            'timeline_raw.png', 'timeline_log.png', 'histogram_raw.png', 'histogram_log.png'}
