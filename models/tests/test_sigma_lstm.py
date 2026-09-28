"""Small shape, variance-formula, and gradient check for the paper's cell."""

import importlib.util
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
path = root / 'models' / 'sigma-lstm.py'
spec = importlib.util.spec_from_file_location('sigma_lstm', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sigma_lstm():
    torch.manual_seed(42)
    model = module.SigmaLSTM(4)
    from models.base_lstm import FEBCellLSTM, FEBLSTM
    assert isinstance(model, FEBLSTM) and isinstance(model.cell, FEBCellLSTM)
    log_volatility = torch.tensor([[[-3.0], [-2.0]], [[-4.0], [-3.0]]])
    initial_hidden = log_volatility.new_zeros((2, 4))
    initial_memory = torch.ones_like(initial_hidden)
    joined = torch.cat((initial_hidden, log_volatility[:, 0]), dim=1)
    expected_memory = (torch.sigmoid(model.cell.forget_gate(joined)) * initial_memory
                       + torch.sigmoid(model.cell.input_gate(joined))
                       * torch.tanh(model.cell.candidate_gate(joined)))
    torch.manual_seed(42)
    gate_hidden, gate_memory = model.cell(log_volatility[:, 0], initial_hidden, initial_memory)
    torch.manual_seed(42)
    gate_variance = F.softplus(model.cell.output_gate(expected_memory.square()))
    expected_gate = torch.randn_like(gate_variance) * torch.sqrt(
        gate_variance + torch.finfo(gate_variance.dtype).tiny)
    torch.testing.assert_close(gate_memory, expected_memory)
    torch.testing.assert_close(gate_hidden, expected_gate * torch.tanh(expected_memory))
    assert (gate_variance >= 0).all()

    torch.manual_seed(42)
    predicted_log_volatility, predicted_log_volatility_variance = model(log_volatility)
    assert predicted_log_volatility.shape == predicted_log_volatility_variance.shape == (2, 1)
    assert torch.isfinite(predicted_log_volatility_variance).all()
    assert (predicted_log_volatility_variance >= 0).all()
    assert (torch.exp(2 * predicted_log_volatility) > 0).all()

    model.capture_memory = True
    torch.manual_seed(42)
    traced_log_volatility, _, memory_trace = model(log_volatility)
    model.capture_memory = False
    torch.testing.assert_close(traced_log_volatility, predicted_log_volatility)
    assert memory_trace.shape == (2, 2, 4)
    assert torch.isfinite(memory_trace).all()

    torch.manual_seed(42)
    hidden = log_volatility.new_zeros((2, 4))
    memory = torch.ones_like(hidden)
    for time in range(2):
        hidden, memory = model.cell(log_volatility[:, time], hidden, memory)
    torch.testing.assert_close(predicted_log_volatility, model.output_layer(hidden))
    torch.testing.assert_close(predicted_log_volatility_variance, memory.mean(dim=-1, keepdim=True).square())
    torch.testing.assert_close(memory_trace[-1], memory)
    (predicted_log_volatility.square() + predicted_log_volatility_variance).sum().backward()
    assert model.cell.output_gate.weight.grad is not None

    fragile = module.SigmaLSTMCell(1)
    with torch.no_grad():
        fragile.output_gate.weight.fill_(-1000.)
    zero = torch.zeros(1, 1)
    fragile_hidden, _ = fragile(zero, zero, torch.ones_like(zero))
    fragile_hidden.sum().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in fragile.parameters())
    torch.testing.assert_close(fragile.output_gate.weight.grad,
                               torch.zeros_like(fragile.output_gate.weight.grad))
    active = module.SigmaLSTMCell(1)
    with torch.no_grad():
        active.output_gate.weight.fill_(1.)
    torch.manual_seed(42)
    active(zero, zero, torch.ones_like(zero))[0].sum().backward()
    assert torch.isfinite(active.output_gate.weight.grad).all()
    assert active.output_gate.weight.grad.abs().sum() > 0
    negative = module.SigmaLSTMCell(1)
    with torch.no_grad():
        negative.output_gate.weight.fill_(-20.)
    torch.manual_seed(42)
    negative(zero, zero, torch.ones_like(zero))[0].sum().backward()
    assert torch.isfinite(negative.output_gate.weight.grad).all()
    assert negative.output_gate.weight.grad.abs().sum() > 0


def test_sigma_training_contract():
    from models.variance_fit import log_variance_qlike, neural_log_variance
    from models.variance_neural import make_model

    torch.manual_seed(42)
    model = make_model('sigma_lstm', 20, 4, input_size=1)
    log_volatility, auxiliary = model(torch.full((2, 20, 1), 0.5))
    assert log_volatility.shape == auxiliary.shape == (2, 1)
    log_variance = neural_log_variance(log_volatility, 1e-12, 'log_volatility')
    torch.testing.assert_close(log_variance, 2 * log_volatility)
    loss = log_variance_qlike(torch.tensor([[-3.0], [-2.0]]), log_variance, True)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.output_layer.weight.grad is not None


def test_memory_diagnostic_on_nonfinite_output():
    import json
    from io import StringIO
    from contextlib import redirect_stdout
    from tempfile import TemporaryDirectory
    from torch.utils.data import TensorDataset
    from models.variance_fit import fit_variance_network

    model = module.SigmaLSTM(1)
    model.capture_memory = True
    with torch.no_grad():
        model.cell.output_gate.weight.fill_(1000.)
        model.output_layer.weight.fill_(1.)
        model.output_layer.bias.fill_(float('inf'))
    data = TensorDataset(torch.zeros(2, 1, 1), torch.zeros(2, 1))
    settings = {'model': 'sigma_lstm', 'output_convention': 'log_volatility',
                'data_transform': 'log-volatility', 'training_history': {},
                'clip_norm': None}
    with TemporaryDirectory() as directory, redirect_stdout(StringIO()) as log:
        try:
            fit_variance_network(model, data, data, 1e-6, Path(directory) / 'unused.pth',
                                 settings, epochs=1)
        except ValueError as error:
            assert str(error) == 'sigma-LSTM produced nonfinite log volatility'
        else:
            raise AssertionError('expected nonfinite sigma output')
    diagnostic = next(json.loads(line) for line in log.getvalue().splitlines()
                      if '"progress": "nonfinite_sigma_output"' in line)
    assert diagnostic['memory_t_min'] <= diagnostic['memory_t_mean'] <= diagnostic['memory_t_max']
    assert diagnostic['memory_t_finite_count'] == 2
    assert diagnostic['memory_t_nonfinite_count'] == 0


if __name__ == '__main__':
    test_sigma_lstm()
    test_sigma_training_contract()
    test_memory_diagnostic_on_nonfinite_output()
