"""Unit tests for News Impact Curve (NIC) simulation and plotting for MCS models."""

from pathlib import Path
from contextlib import closing
import sqlite3
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd


class TestPlotNewsImpactCurve(unittest.TestCase):
    def test_evaluate_rfsv_baseline(self):
        try:
            from data.plot_news_impact_curve import evaluate_rfsv_baseline
        except ImportError:
            return

        rfsv_params = {
            'AAPL': {'H': 0.1, 'nu_squared': 0.04},
            'AMZN': {'H': 0.12, 'nu_squared': 0.05},
            'GOOG': {'H': 0.11, 'nu_squared': 0.045},
            'NFLX': {'H': 0.15, 'nu_squared': 0.06},
            'NVDA': {'H': 0.13, 'nu_squared': 0.055},
        }
        pred_20 = evaluate_rfsv_baseline(rfsv_params, 20, 1e-5)
        pred_80 = evaluate_rfsv_baseline(rfsv_params, 80, 1e-5)

        self.assertIsInstance(pred_20, float)
        self.assertIsInstance(pred_80, float)
        self.assertTrue(np.isfinite(pred_20) and pred_20 > 0)
        self.assertTrue(np.isfinite(pred_80) and pred_80 > 0)

    def test_saved_artifacts_exist(self):
        csv_path = Path('imgs/data_properties/mcs_news_impact_curve.csv')
        png_path = Path('imgs/data_properties/mcs_news_impact_curve.png')

        self.assertTrue(csv_path.exists(), 'mcs_news_impact_curve.csv missing')
        self.assertTrue(png_path.exists(), 'mcs_news_impact_curve.png missing')

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 11)
        self.assertEqual(list(df['shock_pct']), list(range(-5, 6)))
        np.testing.assert_allclose(df['return'], df['shock_pct'] / 100.0)
        self.assertTrue((df.baseline_variance == 3.396062419686545e-13).all())
        np.testing.assert_allclose(df.baseline_logvol, 0.5 * np.log(df.baseline_variance))

        expected_models = [
            'Base LSTM-20 Global', 'Base LSTM-80 Global', 'Base LSTM-80 Local',
            'MLP-20 Global', 'MLP-80 Global',
            'HARNet-20 Global', 'HARNet-80 Global', 'HARNet-80 Local',
            'RFSV-20 Local', 'RFSV-80 Local',
        ]
        for col in expected_models:
            self.assertIn(col, df.columns)
            self.assertTrue((df[col] > 0).all())
            self.assertTrue(np.isfinite(df[col]).all())

        mean_df = pd.read_csv('imgs/data_properties/mcs_news_impact_curve_mean.csv')
        self.assertTrue(Path('imgs/data_properties/mcs_news_impact_curve_mean.png').exists())
        self.assertEqual(list(mean_df.shock_pct), list(df.shock_pct))
        np.testing.assert_allclose(mean_df.baseline_variance, 0.00099822850495058469)

    def test_training_mean_variance_uses_valid_training_cohort(self):
        from data.plot_news_impact_curve import training_mean_variance

        with TemporaryDirectory() as directory:
            database = Path(directory) / 'history.db'
            with closing(sqlite3.connect(database)) as conn:
                conn.execute('CREATE TABLE raw_history (Ticker TEXT, Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume REAL)')
                for ticker in ('A', 'B'):
                    for i in range(81):
                        high = 3. if i == 0 else 1. if i == 1 else 0.5 if i == 2 else 2.
                        conn.execute('INSERT INTO raw_history VALUES (?, ?, 1, ?, 1, 1, 10)',
                                     (ticker, str((pd.Timestamp('2015-01-01') + pd.Timedelta(days=i)).date()), high))
                conn.execute("INSERT INTO raw_history VALUES ('A', '2025-12-31', 1, 100, 1, 1, 10)")
                conn.commit()
            floor = 0.5 * np.log(2.) ** 2
            mean, count = training_mean_variance(database, floor)
            self.assertEqual(count, 80)
            self.assertAlmostEqual(mean, (79 * floor + 0.5 * np.log(3.) ** 2) / 80)

    def test_simulate_mcs_nic_structure(self):
        try:
            import torch
            from data.plot_news_impact_curve import plot_mcs_nic, simulate_mcs_nic
        except (ImportError, UserWarning):
            # Skip if torch cannot be loaded in this specific environment
            return

        class DummySeqModel(torch.nn.Module):
            def __init__(self, log_val=-9.5):
                super().__init__()
                self.log_val = log_val

            def forward(self, x):
                self.last_input = x.clone()
                return torch.tensor([[self.log_val]])

        class DummyVarModel(torch.nn.Module):
            def __init__(self, var_val=8e-5):
                super().__init__()
                self.var_val = var_val

            def forward(self, x):
                return torch.tensor([[self.var_val]])

        mock_models = {
            'Baseline Variance': 1e-5,
            'Base LSTM-20 Global': DummySeqModel(-9.0),
            'Base LSTM-80 Global': DummySeqModel(-9.2),
            'Base LSTM-80 Local': [DummySeqModel(-9.1) for _ in range(5)],
            'MLP-20 Global': DummySeqModel(-9.3),
            'MLP-80 Global': DummySeqModel(-9.4),
            'HARNet-20 Global': DummyVarModel(8.6e-5),
            'HARNet-80 Global': DummyVarModel(7.3e-5),
            'HARNet-80 Local': [DummyVarModel(8.9e-5) for _ in range(5)],
            'RFSV Parameters': {
                t: {'H': 0.1, 'nu_squared': 0.05}
                for t in ('AAPL', 'AMZN', 'GOOG', 'NFLX', 'NVDA')
            },
        }

        shocks = np.array([-5, 0, 5])
        df = simulate_mcs_nic(mock_models, shock_percents=shocks)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df['shock_pct']), [-5, 0, 5])
        np.testing.assert_allclose(df['return'], [-0.05, 0.0, 0.05])
        np.testing.assert_allclose(df.baseline_logvol, 0.5 * np.log(1e-5))
        x = mock_models['Base LSTM-20 Global'].last_input
        self.assertEqual(tuple(x.shape), (1, 20, 2))
        np.testing.assert_allclose(x[0, :, 0], 0.5 * np.log(1e-5), rtol=1e-6)
        self.assertAlmostEqual(float(x[0, -1, 1]), 0.05, places=6)


if __name__ == '__main__':
    unittest.main()
