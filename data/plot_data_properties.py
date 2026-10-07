"""Plot cross-stock test log variance against preceding returns and rolling means."""

import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.plot_gk_data import LOCAL
from models.support_scripts.raw_neural_data import raw_frame


WINDOW = 252
MODEL = 'MLP-80 GLOBAL'


def preceding_reference(variance, window=WINDOW):
    """Mean of the preceding observed variances, excluding the target."""
    return pd.Series(variance).shift().rolling(window, min_periods=window).mean().to_numpy()


def series(database, forecasts, forecast_reference=False):
    saved = pd.read_csv(forecasts)
    saved = saved.loc[saved.candidate == MODEL, ['ticker', 'date', 'actual', 'predicted']].copy()
    if (len(saved) != 8800 or saved.duplicated(['ticker', 'date']).any()
            or set(saved.ticker) != LOCAL or (saved.groupby('ticker').size() != 1760).any()):
        raise ValueError('expected 1,760 unique MLP-80 Global test keys per stock')
    if (not np.isfinite(saved[['actual', 'predicted']].to_numpy()).all()
            or (saved[['actual', 'predicted']] <= 0).any().any()):
        raise ValueError('saved variances must be finite and positive')
    saved['date'] = pd.to_datetime(saved.date)

    result = {}
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        for ticker in sorted(LOCAL):
            source = pd.read_sql_query('''SELECT Ticker, Date, Open, High, Low, Close, Volume
                FROM raw_history WHERE Ticker = ? AND Date < '2026-01-01' ORDER BY Date''',
                conn, params=(ticker,))
            frame = raw_frame(source)
            observed = frame.loc[frame.Valid, ['Date', 'RawVariance', 'IntradayLogReturn']].copy()
            observed['Date'] = pd.to_datetime(observed.Date)
            train = observed.loc[(observed.Date < '2016-01-01') & (observed.RawVariance > 0), 'RawVariance']
            if train.empty:
                raise ValueError(f'{ticker}: no positive pre-2016 variance')
            observed['variance'] = observed.RawVariance.mask(observed.RawVariance == 0, train.min())
            observed['preceding_return'] = observed.IntradayLogReturn.shift()
            reference_values = observed.variance.copy()
            if forecast_reference:
                ticker_forecasts = saved.loc[saved.ticker == ticker].set_index('date').predicted
                forecast_values = observed.Date.map(ticker_forecasts)
                reference_values = forecast_values.fillna(reference_values)
            observed['reference'] = preceding_reference(reference_values.to_numpy())
            selected = saved.loc[saved.ticker == ticker].merge(
                observed[['Date', 'variance', 'preceding_return', 'reference']],
                left_on='date', right_on='Date', validate='one_to_one', how='left').sort_values('date')
            if (selected.Date.isna().any() or selected[['preceding_return', 'reference']].isna().any().any()
                    or not np.isclose(selected.actual, selected.variance, rtol=1e-10, atol=1e-13).all()
                    or not selected.date.between('2019-01-01', '2025-12-31').all()):
                raise ValueError(f'{ticker}: missing history or saved target mismatch')
            result[ticker] = selected
    return result


def average_stocks(data):
    combined = pd.concat([frame.assign(ticker=ticker) for ticker, frame in data.items()])
    if (set(data) != LOCAL or len(combined) != 8800
            or combined.duplicated(['ticker', 'date']).any()):
        raise ValueError('expected five stocks on 1,760 shared dates')
    grouped = combined.groupby('date', sort=True)
    if len(grouped) != 1760 or (grouped.ticker.nunique() != 5).any():
        raise ValueError('test dates are not shared by all five stocks')
    result = grouped[['actual', 'predicted', 'preceding_return', 'reference']].mean()
    if (not np.isfinite(result.to_numpy()).all()
            or (result[['actual', 'predicted', 'reference']] <= 0).any().any()):
        raise ValueError('cross-stock means must be finite and positive')
    return result


def binned_curve(frame, bin_size=20):
    """Average consecutive return-sorted pairs in groups of bin_size."""
    values = pd.DataFrame({'return': frame.preceding_return,
                           'actual': np.log(frame.actual),
                           'predicted': np.log(frame.predicted)}).sort_values('return').reset_index(drop=True)
    if len(values) % bin_size:
        raise ValueError('return bins must have equal counts')
    curve = values.groupby(np.arange(len(values)) // bin_size).mean()
    return curve.reset_index(drop=True)


def plot_return_curve(frame, output, model, title):
    curve = binned_curve(frame)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve['return'], curve.actual, marker='o', markersize=3, lw=1.2,
            color='#236c93', label='Actual: 20-return bins')
    ax.plot(curve['return'], curve.predicted, marker='o', markersize=3, lw=1.2,
            color='#bc4b2f', label=f'{model}: 20-return bins')
    ax.set(xlabel='Mean preceding adjusted log(Close/Open)',
           ylabel='Log mean GK variance',
           title=title)
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return curve


def plot(data, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = average_stocks(data)
    curve = plot_return_curve(frame, output / 'volatility_clustering.png', 'MLP-80 Global',
                              'Five-stock mean test variance by mean preceding signed return')
    curve.to_csv(output / 'volatility_clustering_bins.csv', index=False)
    (output / 'volatility_clustering_lowess.csv').unlink(missing_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame.index, np.log(frame.actual / frame.reference), color='#236c93',
            lw=.8, label='Actual')
    ax.plot(frame.index, np.log(frame.predicted / frame.reference), color='#bc4b2f',
            lw=.8, label='MLP-80 Global')
    ax.axhline(0, color='#555555', lw=.6)
    ax.set(xlabel='Test target date', ylabel='Log variance / trailing reference',
           title='Five-stock mean variance relative to preceding 252-observation mean')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'mean_reversion.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('D:/DBs/timeseries_analysis/history_coverage.db'))
    parser.add_argument('--forecasts', type=Path, default=Path('imgs/eval/mcs/80_size/forecasts.csv'))
    parser.add_argument('--output', type=Path, default=Path('imgs/data_properties'))
    args = parser.parse_args()
    data = series(args.database, args.forecasts)
    plot(data, args.output)
    print({ticker: {'targets': len(frame), 'first': str(frame.date.iloc[0].date()),
                    'last': str(frame.date.iloc[-1].date())} for ticker, frame in data.items()})


if __name__ == '__main__':
    main()
