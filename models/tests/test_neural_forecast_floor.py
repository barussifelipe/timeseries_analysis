"""Small-variance neural forecast and HARNet floor checks."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from models.training_blocks import floor_prediction, load_fit, qlike
from models.variance_fit import (fit_variance_network, log_variance_qlike,
                                 neural_log_variance)
from models.variance_neural import make_model, neural_predict, neural_train


def test_neural_forecast_floor():
    frame = pd.DataFrame([('A', f'2015-01-0{i}', float(i) * 1e-8) for i in range(1, 5)]
                         + [('A', '2016-01-01', 5e-8)],
                         columns=['Ticker', 'Date', 'Variance'])
    training = {'A': np.array([1e-8, 2e-8, 3e-8, 4e-8])}
    args = SimpleNamespace(window_size=2, epochs=1, batch_size=2, patience=10)
    for kind in ('mlp', 'silu_lstm', 'base_lstm_vol'):
        def inspect(model, *_args, **_kwargs):
            assert _kwargs['patience'] == 10
            assert _args[4]['patience'] == 10
            layer = model.network[-1] if kind == 'mlp' else model.output_layer
            np.testing.assert_allclose(float(layer.bias.detach()), np.log(2.5e-8), rtol=1e-6)
            bound = np.sqrt(6 / (layer.in_features + layer.out_features)) * .01
            assert 0 < float(layer.weight.detach().abs().max()) <= bound
        with patch('models.variance_neural.prepare', return_value=(frame, training, 1e-8)), \
             patch('models.variance_neural.report_counts', return_value={}), \
             patch('models.variance_neural.artifact_path', return_value=Path('unused.pth')), \
             patch('models.variance_neural.wandb_run', return_value=None), \
             patch('models.variance_fit.fit_variance_network', side_effect=inspect):
            neural_train(kind, args)

    actual = torch.tensor([[1e-8], [1.00001e-8]], dtype=torch.float64)
    predicted = torch.tensor([[1.00001e-8], [1e-8]], dtype=torch.float64)
    np.testing.assert_allclose(float(qlike(actual, predicted)),
                               qlike(actual.numpy(), predicted.numpy()), rtol=1e-10, atol=1e-15)
    extreme_z = torch.tensor([[100.]], requires_grad=True)
    z = neural_log_variance(extreme_z, 1e-10, 'log_variance')
    loss = log_variance_qlike(torch.ones_like(z), z)
    assert torch.isfinite(loss) and loss.item() == 99
    loss.backward()
    assert torch.isfinite(extreme_z.grad).all()
    try:
        neural_log_variance(torch.tensor([[float('nan')]]), 1e-10, 'log_variance')
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite log-variance accepted')
    for kind in ('mlp', 'silu_lstm', 'base_lstm_vol'):
        model = make_model(kind, 20, 8)
        output_layer = model.network[-1] if kind == 'mlp' else model.output_layer
        with torch.no_grad():
            output_layer.weight.mul_(.01)
            output_layer.bias.fill_(np.log(1e-8))
        x = torch.full((2, 20, 1), 1e-8)
        z = neural_log_variance(model(x), 1e-10, 'log_variance')
        loss = log_variance_qlike(torch.tensor([[1e-8], [2e-8]]), z)
        assert torch.isfinite(loss)
        loss.backward()
        assert output_layer.bias.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for name, p in model.named_parameters() if not name.startswith('network.6')
                   and not name.startswith('output_layer'))

    for kind, window in (('harnet_20', 20), ('harnet_80', 80)):
        model = make_model(kind, window)
        coefficients = np.r_[1e-4, np.ones(3 if window == 20 else 5) * .2]
        model.initialize_from_har(coefficients)
        x = torch.full((1, window, 1), 1e-4)
        raw = model(x)
        assert raw.item() > 1e-5
        np.testing.assert_allclose(float(neural_log_variance(raw, 1e-5, 'floored_variance').exp().detach()),
                                   raw.item(), rtol=1e-6)
        with torch.no_grad():
            model.output.bias.fill_(-1.)
            model.output.weight.zero_()
        raw = model(x)
        raw.retain_grad()
        loss = log_variance_qlike(torch.full_like(raw, 1e-4),
                                  neural_log_variance(raw, 1e-5, 'floored_variance'))
        loss.backward()
        assert raw.grad.item() == 0
        class Log:
            def __init__(self):
                self.rows = []
            def log(self, row):
                self.rows.append(row)
        log = Log()
        with tempfile.TemporaryDirectory() as directory:
            data = TensorDataset(x, torch.full((1, 1), 1e-4))
            data.ticker_names, data.ticker_ids = ['A'], torch.tensor([0])
            fit_variance_network(model, data, data, 1e-5,
                                 Path(directory) / 'fit.pth', {'output_convention': 'floored_variance',
                                 'training_history': {'A': [1e-4, 2e-4]}},
                                 epochs=1, run=log)
        assert log.rows[0]['train/floor_hits_pct'] == 100

    with tempfile.TemporaryDirectory() as directory:
        model = make_model('mlp', 2, 16)
        with torch.no_grad():
            model.network[-1].weight.zero_()
            model.network[-1].bias.fill_(np.log(1e-8))
        fit = {'model_state_dict': model.state_dict(), 'floor': 1e-7,
               'output_convention': 'log_variance',
               'settings': {'model': 'mlp', 'window_size': 2, 'output_convention': 'log_variance'}}
        path = Path(directory) / 'fit.pth'
        torch.save(fit, path)
        frame = pd.DataFrame([('A', f'2019-01-0{i}', 1e-8) for i in range(1, 4)],
                             columns=['Ticker', 'Date', 'Variance'])
        result = neural_predict('mlp', load_fit(path), frame)
        np.testing.assert_allclose(result[0].predicted_variance, 1e-8, rtol=1e-6)
        with torch.no_grad():
            model.network[-1].bias.fill_(100.)
        fit['model_state_dict'] = model.state_dict()
        result = neural_predict('mlp', fit, frame)
        np.testing.assert_allclose(result[0].predicted_variance, np.exp(100), rtol=1e-6)
        with torch.no_grad():
            model.network[-1].bias.fill_(800.)
        fit['model_state_dict'] = model.state_dict()
        try:
            neural_predict('mlp', fit, frame)
        except ValueError:
            pass
        else:
            raise AssertionError('nonfinite converted variance accepted at inference')
        fit['output_convention'] = 'floored_variance'
        try:
            neural_predict('mlp', fit, frame)
        except ValueError:
            pass
        else:
            raise AssertionError('incompatible checkpoint accepted')


if __name__ == '__main__':
    test_neural_forecast_floor()
    print('neural forecast floor checks passed')
