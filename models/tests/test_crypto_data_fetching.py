import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from data.crypto_data_fetching import (
    _create_crypto_tables,
    _rebuild_daily_volatility,
    _update_coverage,
    crypto_history_summary,
)


class CryptoDataFetchingTest(unittest.TestCase):
    def test_complete_intraday_day_builds_one_aligned_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'coverage.db'
            conn = sqlite3.connect(database)
            _create_crypto_tables(conn)
            conn.execute("INSERT INTO crypto_symbols VALUES ('BTC', 'Bitcoin', 1)")
            conn.executemany(
                'INSERT INTO crypto_daily_history VALUES (?, ?, ?, ?, ?, ?, ?)',
                [
                    ('BTC', '2019-01-01T00:00:00Z', 1, 1, 1, 1, 1),
                    ('BTC', '2019-01-02T00:00:00Z', 1, 1, 1, 1, 1),
                    ('BTC', '2019-01-03T00:00:00Z', 1, 1, 1, 1, 1),
                ],
            )
            start = datetime(2019, 1, 1, tzinfo=timezone.utc)
            rows = [('BTC', '2018-12-31T23:45:00+00:00', 1, 1, 1, 1, 1)]
            for index in range(192):
                timestamp = start + timedelta(minutes=15 * index)
                rows.append(('BTC', timestamp.isoformat(), 1, 1, 1, 1, 1))
            rows.append(('BTC', '2019-01-03T00:00:00Z', 1, 1, 1, 1, 1))
            conn.executemany(
                'INSERT INTO crypto_intraday_history VALUES (?, ?, ?, ?, ?, ?, ?)',
                rows,
            )

            _rebuild_daily_volatility(conn, 'BTC')
            _update_coverage(
                conn, 'BTC', '2019-01-01T00:00:00Z', '2019-01-03T00:00:00Z'
            )
            conn.commit()
            conn.close()

            result = crypto_history_summary(database)

        self.assertEqual(result['daily_before_na'], 2)
        self.assertEqual(result['daily_after_na'], 2)
        self.assertEqual(result['intraday_days_before_na'], 2)
        self.assertEqual(result['intraday_days_after_na'], 2)
        self.assertEqual(result['aligned_days_after_na'], 2)
        self.assertEqual(result['full_period_symbols'], 1)
        self.assertFalse(result['enough_aligned_points'])


if __name__ == '__main__':
    unittest.main()
