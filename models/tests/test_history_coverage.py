import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from data.data_fetching import (
    HISTORY_COLUMNS,
    _create_history_tables,
    _download_failure_status,
    _period_coverage,
    history_lengths,
)


class HistoryCoverageTest(unittest.TestCase):
    def test_download_failures_remain_distinct(self):
        self.assertEqual(_download_failure_status('HTTP 429: Too Many Requests'), 'rate_limited')
        self.assertEqual(_download_failure_status('possibly delisted; no price data'), 'empty')
        self.assertEqual(_download_failure_status('ConnectionError: timed out'), 'transient')
        self.assertEqual(_download_failure_status('connection refused'), 'failed')

    def test_period_boundary_and_missing_values(self):
        index = pd.to_datetime(['2006-01-03', '2025-12-31'])
        complete = pd.DataFrame(1.0, index=index, columns=HISTORY_COLUMNS)

        self.assertEqual(
            _period_coverage(complete, '20y', pd.Timestamp('2026-01-01'))[:2],
            (True, True),
        )

        complete.loc[index[-1], 'Volume'] = None
        self.assertEqual(
            _period_coverage(complete, '20y', pd.Timestamp('2026-01-01'))[:2],
            (True, False),
        )

    def test_failed_status_is_not_counted_as_history(self):
        with tempfile.TemporaryDirectory() as directory:
            db_filename = Path(directory) / 'coverage.db'
            with closing(sqlite3.connect(db_filename)) as conn, conn:
                _create_history_tables(conn)
                conn.executemany(
                    'INSERT INTO history_downloads (ticker, status) VALUES (?, ?)',
                    [('GOOD', 'success'), ('LIMITED', 'rate_limited')],
                )
                conn.execute(
                    '''
                    INSERT INTO history_coverage
                    VALUES ('GOOD', '20y', 1, 1, 2, 2)
                    '''
                )

            result = history_lengths('20y', db_filename)

        self.assertEqual(result['requested'], 2)
        self.assertEqual(result['after_na'], 1)
        self.assertEqual(result['rate_limited'], 1)


if __name__ == '__main__':
    unittest.main()
