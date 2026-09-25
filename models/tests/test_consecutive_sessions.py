"""Missing ticker sessions must not turn a one-day target into a longer horizon."""

import pandas as pd
import numpy as np

from models.training_blocks import TimeSeriesDataset, report_counts


def test_consecutive_sessions():
    dates = ('2015-12-29', '2015-12-30', '2015-12-31', '2016-01-04', '2016-01-05')
    frame = pd.DataFrame(
        [(ticker, date, 0.01 + i * 0.001)
         for ticker in ('A', 'B') for i, date in enumerate(dates)
         if ticker == 'A' or date != '2015-12-31'],
        columns=('Ticker', 'Date', 'Variance'))
    assert report_counts(frame, 2)['val'] == 4
    assert report_counts(frame, 2, consecutive_sessions=True) == {'train': 1, 'val': 2, 'test': 0}
    train = TimeSeriesDataset(frame, 2, split='train', consecutive_sessions=True)
    val = TimeSeriesDataset(frame, 2, split='val', consecutive_sessions=True)
    assert len(train) == 1 and len(val) == 2
    assert [str(date.date()) for date in val.target_dates] == ['2016-01-04', '2016-01-05']
    assert [val.ticker_names[i] for i in val.ticker_ids.tolist()] == ['A', 'A']
    np.testing.assert_allclose([x.flatten().tolist() for x, _ in val],
                               [[0.011, 0.012], [0.012, 0.013]])
    only_b = frame[frame.Ticker == 'B']
    assert len(TimeSeriesDataset(only_b, 2, split='val')) == 2
    assert len(TimeSeriesDataset(only_b, 2, split='val', consecutive_sessions=True,
                                 session_calendar=dates)) == 0


if __name__ == '__main__':
    test_consecutive_sessions()
    print('consecutive session checks passed')
