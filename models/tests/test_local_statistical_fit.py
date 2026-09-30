"""Known lag counts and train-only local roughness artifact check."""

from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from data.roughness_analysis import local_raw_gk_roughness


def test_local_roughness():
    rng = np.random.default_rng(42)
    values = np.empty(900)
    values[0] = 0
    for i in range(1, len(values)):
        values[i] = .5 * values[i - 1] + rng.normal(scale=.1)
    frame = pd.DataFrame({'Ticker': ['NVDA'] * 901,
                          'Date': pd.date_range('2010-01-01', periods=900).append(
                              pd.DatetimeIndex(['2016-01-01'])),
                          'Valid': True,
                          'RawVariance': np.exp(2 * np.r_[values, 100.])})
    with TemporaryDirectory() as directory:
        fit = local_raw_gk_roughness(frame, 'NVDA', directory)
        moments = pd.read_csv(f'{directory}/roughness_moments.csv')
        summary = pd.read_csv(f'{directory}/roughness_summary.csv')
        assert moments.loc[(moments.Lag == 400) & (moments.q == 2), 'Observations'].item() == 500
        expected = np.mean(np.abs(np.diff(values)) ** 2)
        assert np.isclose(moments.loc[(moments.Lag == 1) & (moments.q == 2), 'Moment'].item(), expected)
        assert np.isclose(fit['H'], summary.H.item())
        assert np.isclose(fit['nu_squared'], summary.NuSquared.item())
        assert (pd.read_csv(f'{directory}/roughness_zeta.csv').Population == 'NVDA').all()


if __name__ == '__main__':
    test_local_roughness()
