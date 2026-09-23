import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.crypto_data_fetching import _create_crypto_tables
from data.gk_histograms import distributions
from data.roughness_analysis import (
    CRYPTO_LENGTH,
    analyze_training_roughness,
    garman_klass_variance,
    parkinson_variance,
    plot_scaling,
    rebuild_variance_tables,
    roughness_moments,
    scaling_estimates,
)


class RoughnessAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        self.conn.executescript('''
            CREATE TABLE raw_history (
                Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL,
                Close REAL, Volume REAL, PRIMARY KEY (Ticker, Date)
            ) WITHOUT ROWID;
            CREATE TABLE history_downloads (
                ticker TEXT PRIMARY KEY, active INTEGER, status TEXT,
                attempts INTEGER DEFAULT 0, first_date TEXT, last_date TEXT,
                last_error TEXT, updated_at TEXT
            );
            CREATE TABLE history_coverage (
                ticker TEXT, period TEXT, before_na INTEGER, after_na INTEGER,
                rows_before_na INTEGER, rows_after_na INTEGER,
                PRIMARY KEY (ticker, period)
            );
        ''')
        _create_crypto_tables(self.conn)

    def tearDown(self):
        self.conn.close()

    def _equity(self, ticker, dates, prices):
        self.conn.execute(
            "INSERT INTO history_downloads (ticker, active, status) "
            "VALUES (?, 1, 'success')",
            (ticker,),
        )
        self.conn.execute(
            "INSERT INTO history_coverage VALUES (?, '25y', 1, 1, ?, ?)",
            (ticker, len(dates), len(dates)),
        )
        self.conn.executemany(
            'INSERT INTO raw_history VALUES (?, ?, ?, ?, ?, ?, ?)',
            ((ticker, date, *price) for date, price in zip(dates, prices)),
        )

    def _crypto(
        self, ticker, days=CRYPTO_LENGTH, start=np.datetime64('2019-01-01')
    ):
        daily = []
        realized = []
        for index in range(days):
            date = str(start + np.timedelta64(index, 'D'))
            daily.append(
                (ticker, f'{date}T00:00:00Z', 10, 12, 9, 11, 100 + index)
            )
            realized.append((ticker, date, .01 + index / 1_000_000, 96))
        self.conn.executemany(
            'INSERT INTO crypto_daily_history VALUES (?, ?, ?, ?, ?, ?, ?)',
            daily,
        )
        self.conn.executemany(
            'INSERT INTO crypto_daily_volatility VALUES (?, ?, ?, ?)',
            realized,
        )

    def test_formulas_match_hand_calculation(self):
        self.assertAlmostEqual(
            parkinson_variance(12, 9),
            math.log(4 / 3) ** 2 / (4 * math.log(2)),
        )
        expected = (
            .5 * math.log(4 / 3) ** 2
            - (2 * math.log(2) - 1) * math.log(1.1) ** 2
        )
        self.assertAlmostEqual(
            garman_klass_variance(10, 12, 9, 11), expected
        )

    def test_gk_histogram_pairs_use_calendar_lags_within_tickers(self):
        self.conn.execute('CREATE TABLE equity_garman_klass_variance '
                          '(Ticker TEXT, Date TEXT, Variance REAL)')
        self.conn.executemany('INSERT INTO equity_garman_klass_variance VALUES (?, ?, ?)', [
            ('A', '2020-01-01', 1.), ('A', '2020-01-02', math.exp(2)),
            ('A', '2020-01-06', math.exp(6)),
            ('B', '2020-01-01', math.exp(20)),
            ('B', '2020-01-02', math.exp(24)),
        ])
        variance, increments = distributions(self.conn)
        self.assertEqual(len(variance), 5)
        np.testing.assert_allclose(increments[1], [1, 2])
        np.testing.assert_allclose(increments[5], [3])
        self.assertEqual(len(increments[25]), 0)

    def test_rebuild_filters_and_is_idempotent(self):
        self._equity(
            'AAA',
            ['2001-01-01', '2001-01-02', '2001-01-03'],
            [
                (10, 12, 9, 11, 100),
                (10, 10, 10, 10, 100),
                (10, 9, 8, 11, 100),
            ],
        )
        self._crypto('BTC')
        self._crypto('USD')
        self._crypto('SHORT', CRYPTO_LENGTH - 1)
        first = rebuild_variance_tables(self.conn)
        keys = [
            self.conn.execute(
                f'SELECT Ticker, Date FROM {table} ORDER BY Ticker, Date'
            ).fetchall()
            for table in (
                'crypto_parkinson_variance',
                'crypto_garman_klass_variance',
                'crypto_realized_variance',
            )
        ]
        second = rebuild_variance_tables(self.conn)

        self.assertEqual(first, second)
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(keys[1], keys[2])
        self.assertEqual(len(keys[0]), CRYPTO_LENGTH)
        self.assertEqual({ticker for ticker, _ in keys[0]}, {'BTC'})
        self.assertEqual(keys[0][0][1], '2019-01-01')
        self.assertEqual(first['equity_parkinson_variance'], 1)

    def test_latest_crypto_dates_are_selected(self):
        self._crypto('BTC', CRYPTO_LENGTH + 2)
        rebuild_variance_tables(self.conn)
        first, last, count = self.conn.execute(
            'SELECT MIN(Date), MAX(Date), COUNT(*) '
            'FROM crypto_realized_variance'
        ).fetchone()
        self.assertEqual(
            (first, last, count), ('2019-01-03', '2023-10-28', CRYPTO_LENGTH)
        )

    def test_moments_never_cross_tickers_and_match_manual_pool(self):
        self.conn.execute(
            'CREATE TABLE sample ('
            'Ticker TEXT, Date TEXT, Variance REAL, '
            'PRIMARY KEY (Ticker, Date)) WITHOUT ROWID'
        )
        self.conn.executemany('INSERT INTO sample VALUES (?, ?, ?)', [
            ('A', '2020-01-01', 1),
            ('A', '2020-01-02', math.exp(2)),
            ('B', '2020-01-03', math.exp(8)),
            ('B', '2020-01-05', math.exp(12)),
        ])
        moments = roughness_moments(
            self.conn, 'sample', lags=[1, 2], qs=[1, 2]
        )
        lag1 = moments[moments.Lag == 1]
        lag2 = moments[moments.Lag == 2]
        self.assertEqual(lag1.Observations.unique().tolist(), [1])
        self.assertEqual(lag2.Observations.unique().tolist(), [1])
        self.assertAlmostEqual(
            lag1.loc[lag1.q == 1, 'Moment'].item(), 1
        )
        self.assertAlmostEqual(
            lag2.loc[lag2.q == 2, 'Moment'].item(), 4
        )

    def test_zero_displacement_and_unavailable_lags(self):
        self.conn.execute(
            'CREATE TABLE sample (Ticker TEXT, Date TEXT, Variance REAL)'
        )
        self.conn.executemany('INSERT INTO sample VALUES (?, ?, ?)', [
            ('A', '2020-01-01', 1),
            ('A', '2020-01-02', 1),
        ])
        moments = roughness_moments(
            self.conn, 'sample', lags=[1, 4], qs=[1]
        )
        self.assertEqual(
            moments.loc[moments.Lag == 1, 'Moment'].item(), 0
        )
        self.assertTrue(
            math.isnan(moments.loc[moments.Lag == 4, 'Moment'].item())
        )
        zeta, hurst, r2 = scaling_estimates(moments)
        self.assertTrue(zeta.Zeta.isna().all())
        self.assertTrue(zeta.Intercept.isna().all())
        self.assertTrue(math.isnan(hurst))
        self.assertTrue(math.isnan(r2))

    def test_training_moments_stop_before_2016(self):
        self.conn.execute('CREATE TABLE sample (Ticker TEXT, Date TEXT, Variance REAL)')
        self.conn.executemany('INSERT INTO sample VALUES (?, ?, ?)', [
            ('A', '2015-12-30', 1.),
            ('A', '2015-12-31', math.exp(2)),
            ('A', '2016-01-01', math.exp(20)),
        ])
        train = roughness_moments(self.conn, 'sample', lags=[1], qs=[1],
                                  end_date='2016-01-01')
        full = roughness_moments(self.conn, 'sample', lags=[1], qs=[1])
        self.assertEqual(train.Observations.item(), 1)
        self.assertEqual(train.Moment.item(), 1.)
        self.assertEqual(full.Observations.item(), 2)

    def test_observation_lags_respect_tickers_cutoff_and_intercept(self):
        self.conn.execute('CREATE TABLE sample (Ticker TEXT, Date TEXT, Variance REAL)')
        self.conn.executemany('INSERT INTO sample VALUES (?, ?, ?)', [
            ('A', '2015-12-28', 1.), ('A', '2015-12-30', math.exp(2)),
            ('A', '2015-12-31', math.exp(6)), ('A', '2016-01-01', math.exp(100)),
            ('B', '2015-12-28', math.exp(20)), ('B', '2015-12-31', math.exp(24)),
        ])
        moments = roughness_moments(self.conn, 'sample', lags=[1, 2], qs=[2],
                                    end_date='2016-01-01', lag_type='observation')
        self.assertEqual(moments.Observations.tolist(), [3, 1])
        self.assertAlmostEqual(moments.Moment.iloc[0], (1 + 4 + 4) / 3)
        self.assertAlmostEqual(moments.Moment.iloc[1], 9)
        zeta, _, _ = scaling_estimates(moments)
        self.assertAlmostEqual(zeta.Intercept.item(), math.log(3))

    def test_training_global_outputs(self):
        for table in ('equity_parkinson_variance', 'equity_garman_klass_variance'):
            self.conn.execute(f'CREATE TABLE {table} (Ticker TEXT, Date TEXT, Variance REAL)')
            self.conn.executemany(f'INSERT INTO {table} VALUES (?, ?, ?)', [
                ('A', f'2015-12-{day:02d}', math.exp((day - 25) ** 2))
                for day in range(26, 31)
            ] + [('A', '2016-01-01', math.exp(50))])
        with tempfile.TemporaryDirectory() as directory:
            summary = analyze_training_roughness(self.conn, directory, max_lag=3)
            output = Path(directory) / 'global' / 'train'
            self.assertEqual(len(summary), 2)
            self.assertTrue(np.isfinite(summary.H).all())
            for name in ('roughness_summary.csv', 'roughness_moments.csv',
                         'roughness_zeta.csv', 'global_hurst.png',
                         'equity_parkinson_variance_global_scaling.png',
                         'equity_garman_klass_variance_global_scaling.png'):
                self.assertTrue((output / name).is_file(), name)

    def test_scaling_figure_is_created(self):
        self.conn.execute(
            'CREATE TABLE sample (Ticker TEXT, Date TEXT, Variance REAL)'
        )
        self.conn.executemany('INSERT INTO sample VALUES (?, ?, ?)', [
            ('A', '2020-01-01', 1),
            ('A', '2020-01-02', math.exp(2)),
            ('A', '2020-01-03', math.exp(6)),
        ])
        moments = roughness_moments(
            self.conn, 'sample', lags=[1, 2], qs=[1, 2]
        )
        zeta, hurst, r2 = scaling_estimates(moments)
        with tempfile.TemporaryDirectory() as directory:
            plot_scaling(
                moments, zeta, hurst, r2, 'Fixture', Path(directory), 'fixture'
            )
            self.assertTrue((Path(directory) / 'fixture_scaling.png').is_file())


if __name__ == '__main__':
    unittest.main()
