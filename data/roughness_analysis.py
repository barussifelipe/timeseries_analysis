"""Build volatility estimators and run calendar-lag roughness analysis."""

import argparse
import math
import os
import sqlite3
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.crypto_data_fetching import HISTORY_DB


EQUITY_START = '2001-01-01'
END_DATE = '2025-12-31'
CRYPTO_START = '2019-01-01'
CRYPTO_LENGTH = 1_760
EXCLUDED_CRYPTO = {
    'BRL', 'CAD', 'DAI', 'EUR', 'GBP', 'JPY', 'MXN', 'PAX', 'TRY', 'TUSD',
    'USD', 'USDC', 'USDT',
}
QS = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
TABLES = {
    'equity_parkinson_variance': ('equity', 'Parkinson'),
    'equity_garman_klass_variance': ('equity', 'Garman-Klass'),
    'crypto_parkinson_variance': ('crypto', 'Parkinson'),
    'crypto_garman_klass_variance': ('crypto', 'Garman-Klass'),
    'crypto_realized_variance': ('crypto', 'Realized variance'),
}


def parkinson_variance(high, low):
    return math.log(high / low) ** 2 / (4 * math.log(2))


def garman_klass_variance(open_, high, low, close):
    return (
        0.5 * math.log(high / low) ** 2
        - (2 * math.log(2) - 1) * math.log(close / open_) ** 2
    )


def _create_variance_table(conn, table):
    conn.execute(f'''
        CREATE TABLE {table} (
            Ticker TEXT NOT NULL,
            Date TEXT NOT NULL,
            Variance REAL NOT NULL CHECK (Variance > 0),
            PRIMARY KEY (Ticker, Date)
        ) WITHOUT ROWID
    ''')


def rebuild_variance_tables(conn):
    """Atomically replace all five derived tables and return rejection counts."""
    conn.create_function(
        'ISFINITE', 1,
        lambda value: value is not None and math.isfinite(value),
    )
    tables = tuple(TABLES)
    try:
        conn.execute('SAVEPOINT rebuild_variance_tables')
        for table in tables:
            conn.execute(f'DROP TABLE IF EXISTS {table}')
            _create_variance_table(conn, table)

        conn.execute('DROP TABLE IF EXISTS temp.valid_equity')
        conn.execute('''
            CREATE TEMP TABLE valid_equity AS
            SELECT r.Ticker, r.Date, r.Volume,
                   LN(r.High / r.Low) * LN(r.High / r.Low) / (4 * LN(2)) AS Parkinson,
                   0.5 * LN(r.High / r.Low) * LN(r.High / r.Low)
                     - (2 * LN(2) - 1) * LN(r.Close / r.Open) * LN(r.Close / r.Open) AS GK
            FROM raw_history r
            JOIN history_coverage c ON c.ticker = r.Ticker
            JOIN history_downloads d ON d.ticker = r.Ticker
            WHERE c.period = '25y' AND c.after_na = 1
              AND d.active = 1 AND d.status = 'success'
              AND r.Date BETWEEN ? AND ?
              AND ISFINITE(r.Open) AND ISFINITE(r.High) AND ISFINITE(r.Low)
              AND ISFINITE(r.Close) AND ISFINITE(r.Volume)
              AND r.Open > 0 AND r.High > 0 AND r.Low > 0 AND r.Close > 0
              AND r.Volume >= 0 AND r.High >= MAX(r.Open, r.Close)
              AND r.Low <= MIN(r.Open, r.Close)
        ''', (EQUITY_START, END_DATE))
        conn.execute('''
            DELETE FROM valid_equity
            WHERE NOT ISFINITE(Parkinson) OR Parkinson <= 0
               OR NOT ISFINITE(GK) OR GK <= 0
        ''')
        conn.execute('''
            INSERT INTO equity_parkinson_variance
            SELECT Ticker, Date, Parkinson FROM valid_equity ORDER BY Ticker, Date
        ''')
        conn.execute('''
            INSERT INTO equity_garman_klass_variance
            SELECT Ticker, Date, GK FROM valid_equity ORDER BY Ticker, Date
        ''')

        exclusions = ','.join('?' for _ in EXCLUDED_CRYPTO)
        conn.execute('DROP TABLE IF EXISTS temp.positive_crypto')
        conn.execute(f'''
            CREATE TEMP TABLE positive_crypto AS
            WITH clean AS (
                SELECT d.symbol AS Ticker, substr(d.time, 1, 10) AS Date,
                       d.volume AS Volume,
                       LN(d.high / d.low) * LN(d.high / d.low) / (4 * LN(2)) AS Parkinson,
                       0.5 * LN(d.high / d.low) * LN(d.high / d.low)
                         - (2 * LN(2) - 1) * LN(d.close / d.open) * LN(d.close / d.open) AS GK,
                       v.realized_variance AS RV
                FROM crypto_daily_history d
                JOIN crypto_daily_volatility v
                  ON v.symbol = d.symbol AND v.date = substr(d.time, 1, 10)
                WHERE substr(d.time, 1, 10) BETWEEN ? AND ?
                  AND d.symbol NOT IN ({exclusions})
                  AND ISFINITE(d.open) AND ISFINITE(d.high) AND ISFINITE(d.low)
                  AND ISFINITE(d.close) AND ISFINITE(d.volume)
                  AND ISFINITE(v.realized_variance)
                  AND d.open > 0 AND d.high > 0 AND d.low > 0 AND d.close > 0
                  AND d.volume >= 0 AND d.high >= MAX(d.open, d.close)
                  AND d.low <= MIN(d.open, d.close) AND v.realized_variance > 0
            )
            SELECT * FROM clean
            WHERE ISFINITE(Parkinson) AND Parkinson > 0
              AND ISFINITE(GK) AND GK > 0
        ''', (CRYPTO_START, END_DATE, *sorted(EXCLUDED_CRYPTO)))
        conn.execute('DROP TABLE IF EXISTS temp.valid_crypto')
        conn.execute('''
            CREATE TEMP TABLE valid_crypto AS
            WITH eligible AS (
                SELECT Ticker FROM positive_crypto
                GROUP BY Ticker HAVING COUNT(*) >= ?
            ), ranked AS (
                SELECT p.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY p.Ticker ORDER BY p.Date DESC
                       ) AS position
                FROM positive_crypto p JOIN eligible e USING (Ticker)
            )
            SELECT Ticker, Date, Volume, Parkinson, GK, RV
            FROM ranked WHERE position <= ?
        ''', (CRYPTO_LENGTH, CRYPTO_LENGTH))
        for table, column in (
            ('crypto_parkinson_variance', 'Parkinson'),
            ('crypto_garman_klass_variance', 'GK'),
            ('crypto_realized_variance', 'RV'),
        ):
            conn.execute(f'''
                INSERT INTO {table}
                SELECT Ticker, Date, {column}
                FROM valid_crypto ORDER BY Ticker, Date
            ''')

        source_equity = conn.execute('''
            SELECT COALESCE(SUM(c.rows_after_na), 0)
            FROM history_coverage c
            JOIN history_downloads d ON d.ticker = c.ticker
            WHERE c.period = '25y' AND c.after_na = 1
              AND d.active = 1 AND d.status = 'success'
        ''').fetchone()[0]
        counts = {
            table: conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            for table in tables
        }
        source_crypto = conn.execute(f'''
            SELECT COUNT(*)
            FROM crypto_daily_history d
            JOIN crypto_daily_volatility v
              ON v.symbol = d.symbol AND v.date = substr(d.time, 1, 10)
            WHERE substr(d.time, 1, 10) BETWEEN ? AND ?
              AND d.symbol NOT IN ({exclusions})
        ''', (CRYPTO_START, END_DATE, *sorted(EXCLUDED_CRYPTO))).fetchone()[0]
        positive_crypto = conn.execute(
            'SELECT COUNT(*) FROM positive_crypto'
        ).fetchone()[0]
        counts.update({
            'equity_rejected': source_equity - counts['equity_parkinson_variance'],
            'crypto_rejected_invalid': source_crypto - positive_crypto,
            'crypto_valid_not_retained': (
                positive_crypto - counts['crypto_parkinson_variance']
            ),
            'crypto_retained_tickers': conn.execute(
                'SELECT COUNT(DISTINCT Ticker) FROM valid_crypto'
            ).fetchone()[0],
        })
        conn.execute('RELEASE SAVEPOINT rebuild_variance_tables')
    except BaseException:
        conn.execute('ROLLBACK TO SAVEPOINT rebuild_variance_tables')
        conn.execute('RELEASE SAVEPOINT rebuild_variance_tables')
        raise
    print(counts)
    return counts


def top_tickers(conn, asset, limit=10):
    """Rank one asset population by mean raw volume on shared retained keys."""
    if asset == 'equity':
        query = '''
            SELECT v.Ticker, AVG(r.Volume) AS AverageVolume
            FROM equity_parkinson_variance v
            JOIN raw_history r USING (Ticker, Date)
            GROUP BY v.Ticker ORDER BY AverageVolume DESC, v.Ticker LIMIT ?
        '''
    elif asset == 'crypto':
        query = '''
            SELECT v.Ticker, AVG(d.volume) AS AverageVolume
            FROM crypto_parkinson_variance v
            JOIN crypto_daily_history d
              ON d.symbol = v.Ticker AND substr(d.time, 1, 10) = v.Date
            GROUP BY v.Ticker ORDER BY AverageVolume DESC, v.Ticker LIMIT ?
        '''
    else:
        raise ValueError("asset must be 'equity' or 'crypto'")
    return pd.read_sql_query(query, conn, params=(limit,))


def _ticker_series(conn, table, ticker, end_date=None):
    rows = conn.execute(
        f'SELECT Date, Variance FROM {table} WHERE Ticker = ? AND (? IS NULL OR Date < ?) ORDER BY Date',
        (ticker, end_date, end_date),
    ).fetchall()
    if not rows:
        return np.array([], dtype='datetime64[D]'), np.array([])
    dates, variance = zip(*rows)
    return (
        np.asarray(dates, dtype='datetime64[D]'),
        0.5 * np.log(np.asarray(variance)),
    )


def _series_moments(dates, values, lags, qs):
    sums = np.zeros((len(lags), len(qs)))
    counts = np.zeros(len(lags), dtype=np.int64)
    if not len(dates):
        return sums, counts
    for start in range(0, len(lags), 50):
        stop = min(start + 50, len(lags))
        wanted = (
            dates[:, None]
            + lags[start:stop].astype('timedelta64[D]')[None, :]
        )
        positions = np.searchsorted(dates, wanted)
        usable = positions < len(dates)
        safe_positions = np.minimum(positions, len(dates) - 1)
        matched = usable & (dates[safe_positions] == wanted)
        differences = np.abs(values[safe_positions] - values[:, None])
        counts[start:stop] += matched.sum(axis=0)
        for column, q in enumerate(qs):
            sums[start:stop, column] += np.where(
                matched, np.power(differences, q), 0
            ).sum(axis=0)
    return sums, counts


def _observation_moments(values, lags, qs):
    sums = np.zeros((len(lags), len(qs)))
    counts = np.maximum(len(values) - lags, 0)
    for i, lag in enumerate(lags):
        if counts[i]:
            differences = np.abs(values[lag:] - values[:-lag])
            sums[i] = np.power(differences[:, None], qs).sum(axis=0)
    return sums, counts


def roughness_moments(conn, table, tickers=None, lags=range(1, 401), qs=QS,
                      end_date=None, lag_type='calendar'):
    """Pool within-ticker absolute log-volatility displacements."""
    if lag_type not in ('calendar', 'observation'):
        raise ValueError('unknown lag type')
    lags = np.asarray(tuple(lags), dtype=int)
    qs = np.asarray(qs, dtype=float)
    sums = np.zeros((len(lags), len(qs)))
    counts = np.zeros(len(lags), dtype=np.int64)
    tickers = tickers or [
        row[0] for row in conn.execute(
            f'SELECT DISTINCT Ticker FROM {table} ORDER BY Ticker'
        )
    ]
    workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(tickers), workers):
            series = [
                _ticker_series(conn, table, ticker, end_date)
                for ticker in tickers[start:start + workers]
            ]
            futures = [
                executor.submit(_series_moments, dates, values, lags, qs)
                if lag_type == 'calendar' else
                executor.submit(_observation_moments, values, lags, qs)
                for dates, values in series
            ]
            for future in futures:
                ticker_sums, ticker_counts = future.result()
                sums += ticker_sums
                counts += ticker_counts
    records = []
    for row, lag in enumerate(lags):
        for column, q in enumerate(qs):
            moment = sums[row, column] / counts[row] if counts[row] else np.nan
            records.append((int(lag), float(q), float(moment), int(counts[row])))
    return pd.DataFrame(
        records, columns=['Lag', 'q', 'Moment', 'Observations']
    )


def scaling_estimates(moments):
    """Estimate zeta(q), then the origin-constrained zeta(q)=Hq relation."""
    rows = []
    for q, group in moments.groupby('q', sort=True):
        valid = group['Moment'].gt(0) & np.isfinite(group['Moment'])
        x = np.log(group.loc[valid, 'Lag'].to_numpy(dtype=float))
        y = np.log(group.loc[valid, 'Moment'].to_numpy(dtype=float))
        if len(x) < 2 or np.ptp(x) == 0:
            rows.append((q, np.nan, np.nan, np.nan))
            continue
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        total = np.square(y - y.mean()).sum()
        r2 = 1 - np.square(y - fitted).sum() / total if total else np.nan
        rows.append((q, slope, intercept, r2))
    zeta = pd.DataFrame(
        rows, columns=['q', 'Zeta', 'Intercept', 'MomentR2']
    )
    valid = np.isfinite(zeta['Zeta'])
    q = zeta.loc[valid, 'q'].to_numpy()
    values = zeta.loc[valid, 'Zeta'].to_numpy()
    if not len(q) or not np.dot(q, q):
        return zeta, np.nan, np.nan
    hurst = np.dot(q, values) / np.dot(q, q)
    fitted = hurst * q
    # An origin-constrained fit uses uncentered R-squared.
    total = np.square(values).sum()
    r2 = 1 - np.square(values - fitted).sum() / total if total else np.nan
    return zeta, float(hurst), float(r2)


def _save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output / name, dpi=160)
    plt.close(fig)


def plot_dataset_characteristics(conn, table, output):
    frame = pd.read_sql_query(
        f'SELECT Date, Variance FROM {table} ORDER BY Date',
        conn, parse_dates=['Date'],
    )
    daily = frame.groupby('Date')['Variance'].agg(
        Median='median',
        Lower=lambda values: values.quantile(.25),
        Upper=lambda values: values.quantile(.75),
    )
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(daily.index, daily['Median'], linewidth=.8)
    axis.fill_between(daily.index, daily['Lower'], daily['Upper'], alpha=.25)
    axis.set(
        title=f'{TABLES[table][0].title()} {TABLES[table][1]}: daily cross-section',
        ylabel='Variance',
    )
    _save(fig, output / 'datasets', f'{table}_dataset.png')


def plot_top_estimators(conn, asset, tickers, output):
    relevant = [table for table, (kind, _) in TABLES.items() if kind == asset]
    fig, axes = plt.subplots(
        len(relevant), 2, figsize=(13, 3.5 * len(relevant)), squeeze=False
    )
    placeholders = ','.join('?' for _ in tickers)
    for row, table in enumerate(relevant):
        frame = pd.read_sql_query(
            f'''SELECT Ticker, Date, Variance FROM {table}
                WHERE Ticker IN ({placeholders}) ORDER BY Ticker, Date''',
            conn, params=tickers, parse_dates=['Date'],
        )
        for ticker, group in frame.groupby('Ticker', sort=True):
            axes[row, 0].plot(
                group['Date'], .5 * np.log(group['Variance']),
                linewidth=.5, label=ticker,
            )
        axes[row, 0].set_title(TABLES[table][1])
        axes[row, 0].set_ylabel('log volatility')
        axes[row, 1].hist(.5 * np.log(frame['Variance']), bins=80)
        axes[row, 1].set_title(f'{TABLES[table][1]} distribution')
    axes[0, 0].legend(ncol=5, fontsize=7)
    _save(fig, output / 'datasets', f'{asset}_top10_estimators.png')


def _draw_scaling_panels(axes, moments, zeta, hurst):
    for q, group in moments.groupby('q'):
        valid = group['Moment'] > 0
        x = np.log(group.loc[valid, 'Lag'])
        y = np.log(group.loc[valid, 'Moment'])
        fit = zeta.loc[zeta['q'] == q].iloc[0]
        line, = axes[0].plot(
            x, y, alpha=.55,
            label=f'q={q:g}, R²={fit.MomentR2:.3f}',
        )
        axes[0].plot(
            x, fit.Intercept + fit.Zeta * x,
            '--', color=line.get_color(), linewidth=1,
        )
    axes[0].set(xlabel='log(Δ)', ylabel='log m(q, Δ)')
    axes[0].set_xlim(left=0)
    axes[0].legend(fontsize=8)
    q_values = np.r_[0, zeta['q'].to_numpy()]
    zeta_values = np.r_[0, zeta['Zeta'].to_numpy()]
    axes[1].plot(
        q_values, zeta_values, 'o-', label='ζ(q) estimates'
    )
    axes[1].plot(
        q_values, hurst * q_values, '--',
        label=f'Hq fit: H={hurst:.3f}',
    )
    axes[1].set(xlabel='q', ylabel='ζ(q)')
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)


def plot_scaling(moments, zeta, hurst, _hurst_r2, title, output, stem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _draw_scaling_panels(axes, moments, zeta, hurst)
    observations = int(moments.groupby('Lag')['Observations'].first().sum())
    fig.suptitle(
        f'{title} | pooled displacements={observations:,} | '
        f'H={hurst:.3f}'
    )
    _save(fig, output, f'{stem}_scaling.png')


def plot_global_hurst(summary, output):
    global_rows = summary[summary['Population'] == 'global'].copy()
    global_rows['Label'] = global_rows['Table'].map(
        lambda table: f'{TABLES[table][0].title()}\n{TABLES[table][1]}'
    )
    figure, axis = plt.subplots(figsize=(9, 4))
    bars = axis.bar(
        global_rows['Label'], global_rows['H'],
        color=[
            'tab:blue' if TABLES[table][0] == 'equity' else 'tab:orange'
            for table in global_rows['Table']
        ],
    )
    axis.bar_label(bars, labels=[f'{value:.3f}' for value in global_rows['H']])
    axis.set(title='Global Hurst estimates', ylabel='H')
    _save(figure, output, 'global_hurst.png')


def analyze_training_roughness(conn, output='imgs/roughness_analysis', max_lag=400):
    """Save pre-2016 global equity roughness without changing full-period CSVs."""
    output = Path(output) / 'global' / 'train'
    summaries, moments_all, zeta_all = [], [], []
    for table in ('equity_parkinson_variance', 'equity_garman_klass_variance'):
        moments = roughness_moments(conn, table, lags=range(1, max_lag + 1),
                                    end_date='2016-01-01', lag_type='observation')
        zeta, hurst, r2 = scaling_estimates(moments)
        observations = int(moments.groupby('Lag')['Observations'].first().sum())
        summaries.append((table, 'global', observations, hurst, r2, 'observation', '2016-01-01', max_lag))
        moments_all.append(moments.assign(Table=table, Population='global', LagType='observation', TrainEnd='2016-01-01', MaxLag=max_lag))
        zeta_all.append(zeta.assign(Table=table, Population='global', LagType='observation', TrainEnd='2016-01-01', MaxLag=max_lag))
        plot_scaling(moments, zeta, hurst, r2,
                     f'Equity {TABLES[table][1]} (global, pre-2016, observation lags)', output,
                     f'{table}_global')
    summary = pd.DataFrame(summaries, columns=['Table', 'Population', 'Observations', 'H', 'R2', 'LagType', 'TrainEnd', 'MaxLag'])
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / 'roughness_summary.csv', index=False)
    pd.concat(moments_all, ignore_index=True).to_csv(output / 'roughness_moments.csv', index=False)
    pd.concat(zeta_all, ignore_index=True).to_csv(output / 'roughness_zeta.csv', index=False)
    plot_global_hurst(summary, output)
    return summary


def analyze_database(conn, output='imgs/roughness_analysis', max_lag=400):
    """Generate deterministic global/local outputs for all estimator tables."""
    output = Path(output)
    top = {
        asset: top_tickers(conn, asset)['Ticker'].tolist()
        for asset in ('equity', 'crypto')
    }
    for asset, tickers in top.items():
        plot_top_estimators(conn, asset, tickers, output)

    summaries = []
    all_moments = []
    all_zeta = []
    for table, (asset, estimator) in TABLES.items():
        plot_dataset_characteristics(conn, table, output)
        populations = [('global', None)] + [
            (ticker, [ticker]) for ticker in top[asset]
        ]
        for population, tickers in populations:
            moments = roughness_moments(
                conn, table, tickers, range(1, max_lag + 1)
            )
            zeta, hurst, r2 = scaling_estimates(moments)
            observations = int(
                moments.groupby('Lag')['Observations'].first().sum()
            )
            summaries.append((table, population, observations, hurst, r2))
            all_moments.append(
                moments.assign(Table=table, Population=population)
            )
            all_zeta.append(
                zeta.assign(Table=table, Population=population)
            )
            plot_output = (
                output / 'global' / 'full'
                if population == 'global'
                else output / 'local' / asset
            )
            plot_scaling(
                moments, zeta, hurst, r2,
                f'{asset.title()} {estimator} ({population})', plot_output,
                f'{table}_{population}',
            )
    summary = pd.DataFrame(
        summaries, columns=['Table', 'Population', 'Observations', 'H', 'R2']
    )
    csv_output = output / 'csv'
    csv_output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_output / 'roughness_summary.csv', index=False)
    moment_output = pd.concat(all_moments, ignore_index=True)
    moment_output.to_csv(csv_output / 'roughness_moments.csv', index=False)
    pd.concat(all_zeta, ignore_index=True).to_csv(
        csv_output / 'roughness_zeta.csv', index=False
    )
    plot_global_hurst(summary, output / 'global' / 'full')
    analyze_training_roughness(conn, output, max_lag)
    for asset in ('equity', 'crypto'):
        subset = summary[
            summary['Table'].str.startswith(asset)
            & (summary['Population'] != 'global')
        ]
        figure, axis = plt.subplots(figsize=(10, 4))
        for table, group in subset.groupby('Table'):
            axis.plot(
                group['Population'], group['H'], marker='o',
                label=TABLES[table][1],
            )
        axis.set(title=f'{asset.title()} local Hurst estimates', ylabel='H')
        axis.tick_params(axis='x', rotation=45)
        axis.legend()
        _save(figure, output / 'local' / asset, f'{asset}_local_hurst.png')
    return summary


def replot_scaling_from_csv(output='imgs/roughness_analysis'):
    """Refresh scaling plots and zeta fits without recomputing raw moments."""
    output = Path(output)
    csv_output = output / 'csv'
    moments = pd.read_csv(csv_output / 'roughness_moments.csv')
    zetas = []
    for (table, population), group in moments.groupby(
        ['Table', 'Population'], sort=False
    ):
        zeta, hurst, r2 = scaling_estimates(group)
        zetas.append(zeta.assign(Table=table, Population=population))
        asset, estimator = TABLES[table]
        plot_output = (
            output / 'global' / 'full'
            if population == 'global'
            else output / 'local' / asset
        )
        plot_scaling(
            group, zeta, hurst, r2,
            f'{asset.title()} {estimator} ({population})', plot_output,
            f'{table}_{population}',
        )
    pd.concat(zetas, ignore_index=True).to_csv(
        csv_output / 'roughness_zeta.csv', index=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default=HISTORY_DB)
    parser.add_argument('--output', default='imgs/roughness_analysis')
    parser.add_argument('--max-lag', type=int, default=400)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--rebuild-only', action='store_true')
    mode.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args()
    with closing(sqlite3.connect(args.database)) as conn:
        if not args.analysis_only:
            rebuild_variance_tables(conn)
        if not args.rebuild_only:
            analyze_database(conn, args.output, args.max_lag)


if __name__ == '__main__':
    main()
