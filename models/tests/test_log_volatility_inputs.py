"""Log-volatility inputs and targets keep variance-unit scoring."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.training_blocks import TimeSeriesDataset
from models.variance_fit import log_variance_qlike
from models.variance_neural import make_model, model_data, neural_predict


def test_log_volatility_inputs():
    rows = [(ticker, date, float(np.exp(2 * (offset + i))))
            for ticker, offset in (('A', 0), ('B', 4))
            for i, date in enumerate(('2015-01-01', '2015-01-02', '2015-01-03', '2016-01-01'))]
    frame = pd.DataFrame(rows[::-1], columns=('Ticker', 'Date', 'Variance'))
    transformed, feature = model_data(frame, 'log-volatility')
    assert feature == 'LogVolatility'
    np.testing.assert_allclose(transformed.LogVolatility, [7, 6, 5, 4, 3, 2, 1, 0])
    train = TimeSeriesDataset(transformed, 2, feature_columns=(feature,), target_column=feature,
                              split='train', positive_target=False)
    val = TimeSeriesDataset(transformed, 2, feature_columns=(feature,), target_column=feature,
                            split='val', positive_target=False)
    assert len(train) == len(val) == 2
    np.testing.assert_allclose(train[0][0].flatten(), [0, 1])
    np.testing.assert_allclose(train[1][0].flatten(), [4, 5])
    np.testing.assert_allclose(val[0][0].flatten(), [1, 2])
    np.testing.assert_allclose(val[1][0].flatten(), [5, 6])
    np.testing.assert_allclose([val[i][1].item() for i in range(2)], [3, 7])
    y = torch.tensor([[-2.], [3.]])
    z = torch.tensor([[-3.], [5.]])
    torch.testing.assert_close(log_variance_qlike(y, z, True),
                               log_variance_qlike((2 * y).exp(), z))
    model = make_model('base_lstm_vol', 2, 2)
    fit = {'model_state_dict': model.state_dict(), 'floor': 1.,
           'output_convention': 'log_variance',
           'settings': {'model': 'base_lstm_vol', 'window_size': 2, 'hidden_width': 2,
                        'output_convention': 'log_variance', 'data_transform': 'log-volatility'}}
    forecasts = neural_predict('base_lstm_vol', fit, frame, 'val')
    with torch.no_grad():
        expected = [model(x).exp().item() for x, _ in DataLoader(val, batch_size=1)]
    np.testing.assert_allclose([row.predicted_variance for row in forecasts], expected)
    np.testing.assert_allclose([row.actual_variance for row in forecasts], [np.exp(6), np.exp(14)], rtol=1e-6)
    try:
        model_data(frame.assign(Variance=0.), 'log-volatility')
    except ValueError:
        pass
    else:
        raise AssertionError('nonpositive variance accepted before logarithm')


if __name__ == '__main__':
    test_log_volatility_inputs()
