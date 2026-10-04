import numpy as np

from inference.reassess_mcs import aligned_losses


def test_cross_window_alignment():
    dates = (np.datetime64('2019-01-01') + np.arange(1760).astype('timedelta64[D]'))
    rows = [{'date': str(date), 'actual': '1', 'predicted': '1', 'qlike': '0'}
            for date in dates]
    shared = {f'Shared {i}': list(rows) for i in range(6)}
    first = {'AAPL': {**shared, **{f'Window 20 {i}': list(rows) for i in range(12)}}}
    second = {'AAPL': {**shared, **{f'Window 80 {i}': list(rows) for i in range(12)}}}
    names, losses, actual_dates = aligned_losses([first, second], 'AAPL')
    assert len(names) == 30 and losses.shape == (1760, 30)
    assert len(actual_dates) == 1760 and np.all(losses == 0)

    second['AAPL']['Shared 0'] = [dict(row) for row in rows]
    second['AAPL']['Shared 0'][0]['predicted'] = '2'
    second['AAPL']['Shared 0'][0]['qlike'] = str(0.5 + np.log(2) - 1)
    try:
        aligned_losses([first, second], 'AAPL')
    except ValueError as error:
        assert 'different forecasts' in str(error)
    else:
        raise AssertionError('different shared forecasts accepted')
