"""Small known crypto history checks for transfer target selection."""

import sqlite3
import unittest

import numpy as np
import pandas as pd

from inference.evaluate_crypto_global import crypto_frame, selected_dataset
from models.training_blocks import TimeSeriesDataset


class CryptoScoringTest(unittest.TestCase):
    def test_retained_formula_and_prior_only_windows(self):
        conn = sqlite3.connect(':memory:')
        conn.execute('CREATE TABLE crypto_daily_history (symbol TEXT, time TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)')
        conn.execute('CREATE TABLE crypto_garman_klass_variance (Ticker TEXT, Date TEXT, Variance REAL)')
        days = pd.date_range('2020-01-01', periods=1842)
        expected = .5 * np.log(12 / 9)**2 - (2 * np.log(2) - 1) * np.log(11 / 10)**2
        for ticker in ('AAA', 'BBB'):
            conn.executemany('INSERT INTO crypto_daily_history VALUES (?,?,?,?,?,?,?)',
                             [(ticker, str(day.date())+'T00:00:00Z', 10, 12, 9, 11, 2)
                              for day in days])
            conn.executemany('INSERT INTO crypto_garman_klass_variance VALUES (?,?,?)',
                             [(ticker, str(day.date()), expected) for day in days[-1760:]])
        frames = []
        for ticker in ('AAA', 'BBB'):
            frame, keys = crypto_frame(conn, ticker)
            self.assertAlmostEqual(frame.RawVariance.iloc[-1], expected)
            self.assertEqual(conn.execute('SELECT close * volume FROM crypto_daily_history WHERE symbol=? LIMIT 1',
                                          (ticker,)).fetchone()[0], 22)
            for window in (20, 80):
                data = selected_dataset(frame, keys, window)
                self.assertEqual(len(data), 1760)
                self.assertTrue(np.all(data.valid_indices.numpy() + window == data.target_indices.numpy()))
                self.assertEqual(data.target_dates[0], days[82])
            frames.append(frame)
        joined = pd.concat(frames, ignore_index=True)
        data = TimeSeriesDataset(joined, 80, split='test', valid_column='Valid')
        self.assertEqual(len(data), 3524)
        self.assertTrue(np.all(data.valid_indices.numpy()[1762:] >= len(frames[0])))
        conn.close()


if __name__ == '__main__':
    unittest.main()
