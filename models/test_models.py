import math
from unittest.mock import patch

import numpy as np
import torch

from models.base_lstm import FEBLSTM
from models.ar1 import AR1
from models.garch import GARCH
from models.har import HAR
from models.harnet import HARNet
from models.mlp import MLP
from models.rfsv import RFSV
from models.sarima import SARIMA
from models.silu_lstm import SiLULSTM


def test_model_definitions():
    history = np.arange(1., 21.)
    coefficients = np.array([2., 3., 5., 7.])
    expected = 2 + 3 * 20 + 5 * 18 + 7 * 10.5
    assert HAR(coefficients).forecast(history) == expected
    assert AR1(2, 3).forecast(history) == 62

    synthetic = np.arange(1., 41.)
    fitted = HAR().fit(synthetic)
    assert math.isclose(HAR(fitted).forecast(synthetic), 41, abs_tol=1e-10)

    sarima = SARIMA(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    equity = 5 + np.sin(np.arange(80) / 4)
    params = sarima.fit(equity)
    crypto = 8 + np.cos(np.arange(50) / 4)
    with patch.object(SARIMA, "fit", side_effect=AssertionError("forecast refitted")):
        frozen = sarima.forecast(crypto, params)
    assert np.isfinite(frozen)
    assert sarima.forecast(crypto, params) == frozen

    garch = GARCH(1, 0.2, 0.5)
    assert math.isclose(garch.forecast([2.], 4), math.sqrt(3.8))
    for bad in ((0, 0.2, 0.5), (1, -0.1, 0.5), (1, 0.6, 0.5)):
        try:
            GARCH(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid GARCH parameters accepted")

    assert RFSV(H=0.1).forecast(np.ones(20)) == 1
    assert RFSV(H=0.1).forecast(np.linspace(1, 2, 20)) != RFSV(H=0.2).forecast(np.linspace(1, 2, 20))

    x = torch.arange(1., 41.).reshape(2, 20, 1).requires_grad_()
    for model in (MLP(20, hidden_size=8), FEBLSTM(1, 8, 1), SiLULSTM(1, 8), HARNet()):
        output = model(x)
        assert output.shape == (2, 1)
        output.sum().backward(retain_graph=True)
        assert any(p.grad is not None for p in model.parameters())

    harnet = HARNet()
    harnet.initialize_from_har(coefficients)
    np.testing.assert_allclose(harnet(x).detach().flatten(), [expected, HAR(coefficients).forecast(np.arange(21., 41.))], rtol=1e-6)


if __name__ == "__main__":
    test_model_definitions()
    print("model checks passed")
