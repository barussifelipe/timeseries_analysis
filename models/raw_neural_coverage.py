"""Count eligible raw neural windows without materializing the full panel."""
import json
import sqlite3
import sys
from contextlib import closing

import numpy as np
import pandas as pd

from models.raw_neural_data import raw_frame


def coverage(database):
    output = {n: {'train': 0, 'val': 0, 'test': 0, 'zero_targets': 0, 'without_training_history': 0} for n in (20, 80)}
    floor = float('inf')
    missing_training = []
    training_histories = []
    gaps = {'over_7_days': 0, 'over_30_days': 0, 'max_days': 0}
    with closing(sqlite3.connect(database)) as conn:
        names = [row[0] for row in conn.execute("""SELECT Ticker FROM raw_history GROUP BY Ticker
            HAVING SUM(Date < '2016-01-01') > 80 AND SUM(Date = '2025-12-31') > 0""")]
        for name in names:
            rows = pd.read_sql_query("""SELECT Ticker, Date, Open, High, Low, Close, Volume FROM raw_history
                WHERE Ticker = ? AND Date < '2026-01-01' ORDER BY Date""", conn, params=(name,))
            frame = raw_frame(rows)
            dates = frame.Date.to_numpy(dtype='datetime64[D]')
            valid = frame.Valid.to_numpy(dtype=bool)
            variance = frame.RawVariance.to_numpy(dtype=float)
            positive = valid & (variance > 0) & (dates < np.datetime64('2016-01-01'))
            if positive.any():
                floor = min(floor, variance[positive].min())
            elif not (valid & (dates < np.datetime64('2016-01-01'))).any():
                missing_training.append(name)
            training_histories.append((name, variance[valid & (dates < np.datetime64('2016-01-01'))].copy()))
            delta = np.diff(dates).astype('timedelta64[D]').astype(int)
            gaps['over_7_days'] += int((delta > 7).sum())
            gaps['over_30_days'] += int((delta > 30).sum())
            gaps['max_days'] = max(gaps['max_days'], int(delta.max(initial=0)))
            bad = np.r_[0, np.cumsum(~(valid & (variance > 0)))]
            for width in output:
                indices = np.arange(width, len(frame))
                indices = indices[(bad[indices] == bad[indices-width]) & valid[indices]]
                target_dates = dates[indices]
                for split, mask in (('train', target_dates < np.datetime64('2016-01-01')),
                                    ('val', (target_dates >= np.datetime64('2016-01-01')) & (target_dates < np.datetime64('2019-01-01'))),
                                    ('test', (target_dates >= np.datetime64('2019-01-01')) & (target_dates < np.datetime64('2026-01-01')))):
                    output[width][split] += int(mask.sum())
                output[width]['zero_targets'] += int((variance[indices] == 0).sum())
                if name in missing_training:
                    output[width]['without_training_history'] += len(indices)
    invalid_scales = [name for name, raw in training_histories
                      if len(raw) < 2 or np.abs(np.diff(np.where(raw == 0, floor, raw))).mean() <= 0]
    return {'tickers': len(names), 'floor': float(floor), 'gaps': gaps,
            'invalid_mase_tickers': invalid_scales, 'missing_training_tickers': missing_training,
            'windows': output}


if __name__ == '__main__':
    print(json.dumps(coverage(sys.argv[1]), indent=2))
