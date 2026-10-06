"""Check that global references exclude targets and stay within ticker history."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data.plot_global_mean_reversion import target_references


def test_target_references():
    dates = pd.date_range('2018-01-01', periods=254)
    frame = pd.DataFrame({
        'Ticker': pd.Categorical(['A'] * 254 + ['B'] * 254),
        'Date': list(dates) * 2,
        'Valid': [True] * 508,
        'Variance': list(range(1, 255)) + list(range(101, 355)),
    })
    data = SimpleNamespace(target_indices=torch.tensor([252, 253, 506, 507]),
                           ticker_ids=torch.tensor([0, 0, 1, 1]),
                           ticker_names=['A', 'B'])
    result = target_references(frame, data, {'A': 1, 'B': 1})
    np.testing.assert_allclose(result, [126.5, 127.5, 226.5, 227.5])

    short = pd.DataFrame({'Ticker': pd.Categorical(['C'] * 82),
                          'Date': pd.date_range('2019-01-01', periods=82),
                          'Valid': [True] * 82, 'Variance': range(1, 83)})
    one = SimpleNamespace(target_indices=torch.tensor([80]), ticker_ids=torch.tensor([0]),
                          ticker_names=['C'])
    result = target_references(short, one, {'C': 1})
    assert np.isnan(result[0])
