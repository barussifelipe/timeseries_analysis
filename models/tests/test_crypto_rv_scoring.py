"""Known RV day and matched target-window checks for crypto transfer scoring."""

import math
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from data.crypto_data_fetching import _create_crypto_tables, _rebuild_daily_volatility
from inference.evaluate_crypto_global import selected_dataset
from inference.evaluate_crypto_rv import rv_frame


class CryptoRVScoringTest(unittest.TestCase):
    def test_prior_close_enters_96_return_rv(self):
        conn = sqlite3.connect(':memory:')
        _create_crypto_tables(conn)
        start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        times = [start - timedelta(minutes=15)] + [start + timedelta(minutes=15*i) for i in range(96)]
        step = .001
        conn.executemany('INSERT INTO crypto_intraday_history VALUES (?,?,?,?,?,?,?)',
                         [('BTC', time.isoformat(), *(math.exp(k*step),)*4, 1)
                          for k, time in enumerate(times)])
        _rebuild_daily_volatility(conn, 'BTC')
        rows = conn.execute('SELECT date,realized_variance,bars FROM crypto_daily_volatility').fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0][0], rows[0][2]), ('2020-01-01', 96))
        self.assertAlmostEqual(rows[0][1], 96*step**2, places=12)
        conn.close()

    def test_same_targets_with_missing_intervening_rv(self):
        conn = sqlite3.connect(':memory:')
        _create_crypto_tables(conn)
        conn.execute('CREATE TABLE crypto_realized_variance (Ticker TEXT, Date TEXT, Variance REAL)')
        conn.execute('CREATE TABLE crypto_garman_klass_variance (Ticker TEXT, Date TEXT, Variance REAL)')
        dates = pd.date_range('2020-01-01', periods=1844)
        conn.executemany('INSERT INTO crypto_daily_history VALUES (?,?,?,?,?,?,?)',
                         [('BTC', f'{day.date()}T00:00:00Z', 10, 12, 9, 11, 2) for day in dates])
        valid_dates = [day for i, day in enumerate(dates) if i != 1000]
        conn.executemany('INSERT INTO crypto_daily_volatility VALUES (?,?,?,?)',
                         [('BTC', str(day.date()), .01 + i*.00001, 96)
                          for i, day in enumerate(valid_dates)])
        retained = valid_dates[-1760:]
        conn.executemany('INSERT INTO crypto_realized_variance VALUES (?,?,?)',
                         [('BTC', str(day.date()), .01 + (len(valid_dates)-1760+i)*.00001)
                          for i, day in enumerate(retained)])
        conn.executemany('INSERT INTO crypto_garman_klass_variance VALUES (?,?,?)',
                         [('BTC', str(day.date()), .001) for day in retained])
        frame, keys, omitted = rv_frame(conn, 'BTC')
        self.assertEqual(omitted, 1)
        self.assertEqual(len(frame), 1843)
        data20 = selected_dataset(frame, keys, 20)
        data80 = selected_dataset(frame, keys, 80)
        self.assertEqual(len(data20), len(data80), 1760)
        self.assertTrue(np.array_equal(data20.target_dates, data80.target_dates))
        positions = data80.target_indices.numpy()
        self.assertTrue(np.all(data80.valid_indices.numpy() + 80 == positions))
        self.assertEqual(np.count_nonzero(np.diff(frame.Date.iloc[positions].to_numpy(dtype='datetime64[D]'))
                                      > np.timedelta64(1, 'D')), 1)
        conn.close()


if __name__ == '__main__':
    unittest.main()
