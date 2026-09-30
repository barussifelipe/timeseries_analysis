"""Small arithmetic and alignment check for the five-stock comparison."""

import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from inference.evaluate_local_global import CONVENTIONS, aggregate, save_table_png
from inference.evaluate_volatility_test import Totals, scales_from_test
from models.training_blocks import TimeSeriesDataset
from models.variance_neural import model_data


def test_local_global_arithmetic():
    frame = pd.DataFrame({'Ticker': ['A'] * 4, 'Date': pd.to_datetime([
        '2018-12-27', '2018-12-28', '2019-01-02', '2019-01-03']),
        'Variance': [1., 2., 4., 8.], 'RawVariance': [1., 2., 4., 8.],
        'IntradayLogReturn': [.1] * 4, 'Valid': [True] * 4})
    data = TimeSeriesDataset(frame, 1, split='test', valid_column='Valid')
    np.testing.assert_array_equal(data.target_indices.numpy(), [2, 3])
    scale = scales_from_test(frame, data)['A']
    assert scale == 3  # mean of |4-2| and |8-4|, using test target dates
    logged, column = model_data(frame, 'log-volatility')
    assert column == 'LogVolatility'
    np.testing.assert_allclose(logged[column], .5 * np.log(frame.Variance))
    assert CONVENTIONS['mlp'][1] == ('LogVolatility', 'IntradayLogReturn')
    assert CONVENTIONS['silu_lstm'][1] == ('Variance', 'IntradayLogReturn')
    assert CONVENTIONS['harnet_20'][1] == ('Variance',)
    local = Totals()
    global_ = Totals()
    local.add([4, 8], [0, 6], [scale, scale], 1)
    global_.add([4, 8], [5, 9], [scale, scale], 1)
    assert np.isclose(local.result()['MASE'], 5 / 6)
    assert local.result()['floor_hit_pct'] == 50
    assert global_.result()['MASE'] == 1 / 3
    assert local.result()['RMSE'] == np.sqrt((9 + 4) / 2)
    assert aggregate([{'N': 1760, **local.result()}] * 5)['RMSE'] == local.result()['RMSE']
    rows = [{'model': f'Model-{i}', 'population': 'final-date' if i >= 26 else 'full-period',
             'N': 5 if i >= 26 else 8800, 'MAE': float(i + 1), 'MASE': float(i + 1),
             'MSE': float(i + 1), 'RMSE': float(i + 1), 'QLIKE': float(i + 1),
             'floor_hit_pct': 0.} for i in range(28)]
    with TemporaryDirectory() as temporary:
        output = Path(temporary) / 'metrics.png'
        save_table_png(rows, output)
        assert output.is_file() and output.stat().st_size > 0


if __name__ == '__main__':
    test_local_global_arithmetic()
