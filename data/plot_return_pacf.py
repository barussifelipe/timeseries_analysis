"""Plot observed test-date PACF of mean adjusted Open-to-Close returns."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUTS = {
    'Five-stock': Path('imgs/data_properties/local_mlp80_price_reconstruction_gross_return.csv'),
    'Global (including WHLR)': Path('imgs/data_properties/global_base_lstm80_price_reconstruction_gross_return.csv'),
}
OUTPUT = Path('imgs/data_properties')
LAGS = 40


def partial_correlations(values, lags=LAGS):
    """Use sample conditional covariance to remove intervening lags."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) <= 2 * lags:
        raise ValueError('PACF needs a finite one-dimensional series longer than twice the lag count')
    result = np.empty(lags)
    for k in range(1, lags + 1):
        lag_vectors = np.column_stack([x[k - j:len(x) - j] for j in range(k + 1)])
        covariance = np.cov(lag_vectors, rowvar=False)
        endpoints = covariance[np.ix_([0, k], [0, k])]
        if k > 1:
            cross = covariance[np.ix_([0, k], range(1, k))]
            endpoints -= cross @ np.linalg.solve(covariance[1:k, 1:k], cross.T)
        result[k - 1] = endpoints[0, 1] / np.sqrt(endpoints[0, 0] * endpoints[1, 1])
    if not np.isfinite(result).all():
        raise ValueError('undefined partial correlation')
    return result


def main():
    series = {}
    for label, path in INPUTS.items():
        frame = pd.read_csv(path, parse_dates=['date'])
        gross = frame.actual_gross_return.to_numpy(dtype=float)
        if (frame.date.isna().any() or frame.date.duplicated().any()
                or not frame.date.is_monotonic_increasing or len(gross) != 1760
                or not np.isfinite(gross).all() or (gross <= 0).any()):
            raise ValueError(f'{path}: invalid observed gross-return series')
        series[label] = gross
    for suffix, transform, description in (
            ('simple', lambda gross: gross - 1, 'simple return = mean(Close/Open) - 1'),
            ('log', np.log, 'log return = log(mean(Close/Open))')):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for ax, (label, gross) in zip(axes, series.items()):
            coefficients = partial_correlations(transform(gross))
            ax.stem(np.arange(1, LAGS + 1), coefficients, basefmt=' ', linefmt='#236c93', markerfmt='o')
            ax.axhline(0, color='black', linewidth=.7)
            ax.set(ylabel='Partial correlation', title=f'{label} | {len(gross):,} test dates')
            ax.grid(alpha=.2)
        axes[-1].set(xlabel='Lag (trading observations)', xlim=(0, LAGS + 1))
        fig.suptitle(f'Observed adjusted Open-to-Close {description}\nPACF: correlation conditional on intervening lags')
        fig.tight_layout()
        fig.savefig(OUTPUT / f'observed_{suffix}_return_pacf.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
