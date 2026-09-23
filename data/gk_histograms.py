"""Plot full-period global Garman-Klass variance and log-volatility increments."""

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LAGS = (1, 5, 25, 125)  # Calendar days, matching the full-period Hurst analysis.


def distributions(conn):
    """Return pooled raw variances and within-ticker log-volatility increments."""
    variances = []
    increments = {lag: [] for lag in LAGS}
    tickers = conn.execute(
        'SELECT DISTINCT Ticker FROM equity_garman_klass_variance ORDER BY Ticker'
    )
    for (ticker,) in tickers:
        rows = conn.execute(
            'SELECT Date, Variance FROM equity_garman_klass_variance '
            'WHERE Ticker = ? ORDER BY Date', (ticker,)
        ).fetchall()
        dates = np.array([date for date, _ in rows], dtype='datetime64[D]')
        variance = np.array([value for _, value in rows], dtype=float)
        if not np.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError(f'{ticker}: variance must be finite and positive')
        variances.append(variance)
        log_vol = .5 * np.log(variance)
        for lag in LAGS:
            positions = np.searchsorted(dates, dates + np.timedelta64(lag, 'D'))
            valid = positions < len(dates)
            valid[valid] &= dates[positions[valid]] == dates[valid] + np.timedelta64(lag, 'D')
            increments[lag].append(log_vol[positions[valid]] - log_vol[valid])
    return np.concatenate(variances), {
        lag: np.concatenate(parts) for lag, parts in increments.items()
    }


def draw_histogram(axis, values, bounds, title, xlabel, bins=80):
    """Overlay the full-sample normal MLE on an explicitly cropped histogram."""
    mean = values.mean()
    sd = values.std()  # Normal maximum-likelihood estimate (ddof=0).
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f'{title}: normal fit requires nonconstant finite data')
    edges = np.linspace(*bounds, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    width = edges[1] - edges[0]
    axis.bar(edges[:-1], counts / (len(values) * width), width=width,
             align='edge', color='#849ab0', edgecolor='white', linewidth=.25,
             label='Empirical density')
    x = np.linspace(*bounds, 500)
    normal = np.exp(-.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    axis.plot(x, normal, color='#bf3030', linewidth=1.6,
              label=f'Normal MLE: μ={mean:.3g}, σ={sd:.3g}')
    axis.set(title=f'{title} (n={len(values):,})', xlabel=xlabel, ylabel='Density')
    axis.legend(fontsize=8)
    axis.text(.02 if bounds[0] < 0 else .98, .97,
              f'{counts.sum() / len(values):.1%} shown',
              transform=axis.transAxes,
              ha='left' if bounds[0] < 0 else 'right', va='top', fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default='D:/DBs/timeseries_analysis/history_coverage.db')
    parser.add_argument('--output', default='imgs/roughness_analysis/global/full')
    args = parser.parse_args()
    with sqlite3.connect(Path(args.database).resolve().as_uri() + '?mode=ro', uri=True) as conn:
        variance, increments = distributions(conn)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, lag in zip(axes.flat, LAGS):
        values = increments[lag]
        draw_histogram(axis, values, np.percentile(values, [.5, 99.5]),
                       f'Δ = {lag} calendar day' + ('s' if lag != 1 else ''),
                       'log volatility increment')
    fig.suptitle('Global equities: Garman–Klass log-volatility increments, 2001–2025')
    fig.tight_layout()
    fig.savefig(output / 'equity_garman_klass_log_vol_increments.png', dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    draw_histogram(axis, variance, (0, np.percentile(variance, 99.5)),
                   'Global equity Garman–Klass variance, 2001–2025',
                   'Daily raw variance')
    fig.tight_layout()
    fig.savefig(output / 'equity_garman_klass_raw_variance.png', dpi=180)
    plt.close(fig)

    log_variance = np.log(variance)
    fig, axis = plt.subplots(figsize=(10, 5))
    draw_histogram(axis, log_variance, np.percentile(log_variance, [.5, 99.5]),
                   'Global equity Garman–Klass log variance, 2001–2025',
                   'Log daily variance')
    fig.tight_layout()
    fig.savefig(output / 'equity_garman_klass_log_variance.png', dpi=180)
    plt.close(fig)
    print(f'Saved three figures to {output}; variance rows={len(variance):,}; '
          f'lag pairs={[(lag, len(values)) for lag, values in increments.items()]}')


if __name__ == '__main__':
    main()
