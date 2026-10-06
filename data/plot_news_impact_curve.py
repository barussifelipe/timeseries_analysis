"""Simulate News Impact Curves (NIC) for all 10 models in the Model Confidence Set (MCS).

Follows the simulation protocol in reference 1.2 (arXiv:2309.02072, Section 4.3.1):
an input sequence of history length (window=20 or 80) with floor baseline history,
varying the final observation across return shocks s in {-5, ..., +5} percentage points
(s = 100 * log(Close/Open), r = s / 100) to measure asymmetric volatility response (leverage effect).
"""

import argparse
import json
from math import gamma
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betainc
import torch

from models.training_blocks import load_fit
from models.variance_neural import make_model


LOCAL_TICKERS = ('AAPL', 'AMZN', 'GOOG', 'NFLX', 'NVDA')

CHECKPOINT_ROOT = Path('inference/checkpoints')


def load_mcs_models(root=CHECKPOINT_ROOT):
    """Load all 10 models belonging to the MCS intersection."""
    models = {}

    # 1. Base LSTM-20 Global
    p_lstm20 = root / 'base_lstm_vol/garman-klass/global/raw_gk_logvol_return_w20_h128_lr0p001_30e/fit.pth'
    f_lstm20 = load_fit(p_lstm20)
    m_lstm20 = make_model('base_lstm_vol', 20, 128, 2)
    m_lstm20.load_state_dict(f_lstm20['model_state_dict'])
    m_lstm20.eval()
    models['Base LSTM-20 Global'] = m_lstm20
    models['Baseline Variance'] = float(f_lstm20['floor'])

    # 2. Base LSTM-80 Global
    p_lstm80 = root / 'base_lstm_vol/garman-klass/global/base_lstm_raw_gk_logvol_return_w80_h128_lr0p001_20e_wandb/fit.pth'
    f_lstm80 = load_fit(p_lstm80)
    m_lstm80 = make_model('base_lstm_vol', 80, 128, 2)
    m_lstm80.load_state_dict(f_lstm80['model_state_dict'])
    m_lstm80.eval()
    models['Base LSTM-80 Global'] = m_lstm80

    # 3. Base LSTM-80 Local (5 models)
    lstm80_locals = []
    for t in LOCAL_TICKERS:
        p = root / f'base_lstm_vol/garman-klass/local/{t}/base_lstm_local_{t}_raw_gk_logvol_return_w80_h128_lr0p001_20e_qlike/fit.pth'
        f = load_fit(p)
        m = make_model('base_lstm_vol', 80, 128, 2)
        m.load_state_dict(f['model_state_dict'])
        m.eval()
        lstm80_locals.append(m)
    models['Base LSTM-80 Local'] = lstm80_locals

    # 4. MLP-20 Global
    p_mlp20 = root / 'mlp/garman-klass/global/mlp_raw_gk_logvol_return_w20_h128_lr0p001_20e_qlike_online/fit.pth'
    f_mlp20 = load_fit(p_mlp20)
    m_mlp20 = make_model('mlp', 20, 128, 2)
    m_mlp20.load_state_dict(f_mlp20['model_state_dict'])
    m_mlp20.eval()
    models['MLP-20 Global'] = m_mlp20

    # 5. MLP-80 Global
    p_mlp80 = root / 'mlp/garman-klass/global/mlp_raw_gk_logvol_return_w80_h128_lr0p001_20e_qlike_online/fit.pth'
    f_mlp80 = load_fit(p_mlp80)
    m_mlp80 = make_model('mlp', 80, 128, 2)
    m_mlp80.load_state_dict(f_mlp80['model_state_dict'])
    m_mlp80.eval()
    models['MLP-80 Global'] = m_mlp80

    # 6. HARNet-20 Global
    p_harnet20 = root / 'harnet_20/garman-klass/global/harnet_20_raw_gk_variance_w20_lr0p001_20e_qlike_online/fit.pth'
    f_harnet20 = load_fit(p_harnet20)
    m_harnet20 = make_model('harnet_20', 20, 128, 1)
    m_harnet20.load_state_dict(f_harnet20['model_state_dict'])
    m_harnet20.eval()
    models['HARNet-20 Global'] = m_harnet20

    # 7. HARNet-80 Global
    p_harnet80 = root / 'harnet_80/garman-klass/global/harnet_80_raw_gk_variance_w80_lr0p001_20e_qlike_online/fit.pth'
    f_harnet80 = load_fit(p_harnet80)
    m_harnet80 = make_model('harnet_80', 80, 128, 1)
    m_harnet80.load_state_dict(f_harnet80['model_state_dict'])
    m_harnet80.eval()
    models['HARNet-80 Global'] = m_harnet80

    # 8. HARNet-80 Local (5 models)
    harnet80_locals = []
    for t in LOCAL_TICKERS:
        p = root / f'harnet_80/garman-klass/local/{t}/harnet_80_local_{t}_raw_gk_variance_w80_lr0p001_20e_qlike/fit.pth'
        f = load_fit(p)
        m = make_model('harnet_80', 80, 128, 1)
        m.load_state_dict(f['model_state_dict'])
        m.eval()
        harnet80_locals.append(m)
    models['HARNet-80 Local'] = harnet80_locals

    # 9 & 10. RFSV-20 Local and RFSV-80 Local parameter configs
    rfsv_params = {}
    for t in LOCAL_TICKERS:
        p_rfsv = root / f'rfsv/garman-klass/local/{t}/raw_history_full/fit.json'
        rfsv_params[t] = json.loads(p_rfsv.read_text())['parameters']
    models['RFSV Parameters'] = rfsv_params

    return models


def evaluate_rfsv_baseline(rfsv_params, window, base_var):
    """Compute RFSV forecast for constant baseline variance history."""
    edges = np.arange(window + 1, dtype=float)
    preds = []
    for t in LOCAL_TICKERS:
        p = rfsv_params[t]
        h, nu = p['H'], p['nu_squared']
        mass = betainc(0.5 - h, 0.5 + h, edges / (1.0 + edges))
        weights = np.diff(mass)
        weights[-1] += 1.0 - mass[-1]
        constant = 2.0 * gamma(1.5 - h) / (gamma(h + 0.5) * gamma(2.0 - 2.0 * h)) * nu
        preds.append(np.exp(np.log(np.full(window, base_var)) @ weights + constant))
    return float(np.mean(preds))


def simulate_mcs_nic(models, shock_percents=np.arange(-5, 6, 1)):
    """Simulate News Impact Curve for all 10 MCS models across return shocks."""
    base_var = models['Baseline Variance']
    if not np.isfinite(base_var) or base_var <= 0:
        raise ValueError('NIC baseline variance must be finite and positive')
    baseline_logvol = float(0.5 * np.log(base_var))
    rfsv_params = models['RFSV Parameters']
    v_rfsv20 = evaluate_rfsv_baseline(rfsv_params, 20, base_var)
    v_rfsv80 = evaluate_rfsv_baseline(rfsv_params, 80, base_var)

    m_lstm20_g = models['Base LSTM-20 Global']
    m_lstm80_g = models['Base LSTM-80 Global']
    m_lstm80_loc = models['Base LSTM-80 Local']
    m_mlp20_g = models['MLP-20 Global']
    m_mlp80_g = models['MLP-80 Global']
    m_harnet20_g = models['HARNet-20 Global']
    m_harnet80_g = models['HARNet-80 Global']
    m_harnet80_loc = models['HARNet-80 Local']

    # HARNet inputs are constant variance history
    xh20 = torch.full((1, 20), base_var, dtype=torch.float32)
    xh80 = torch.full((1, 80), base_var, dtype=torch.float32)

    with torch.no_grad():
        v_harnet20_g = float(m_harnet20_g(xh20).item())
        v_harnet80_g = float(m_harnet80_g(xh80).item())
        v_harnet80_l = float(np.mean([m(xh80).item() for m in m_harnet80_loc]))

    rows = []
    with torch.no_grad():
        for s in shock_percents:
            r = float(s) / 100.0

            # Sequence inputs: history held at baseline logvol and zero return,
            # final observation holds baseline logvol and return shock r
            x20 = torch.zeros((1, 20, 2), dtype=torch.float32)
            x20[:, :, 0] = baseline_logvol
            x20[0, -1, 1] = r

            x80 = torch.zeros((1, 80, 2), dtype=torch.float32)
            x80[:, :, 0] = baseline_logvol
            x80[0, -1, 1] = r

            v_lstm20_g = float(np.exp(m_lstm20_g(x20).item()))
            v_lstm80_g = float(np.exp(m_lstm80_g(x80).item()))
            v_lstm80_l = float(np.mean([np.exp(m(x80).item()) for m in m_lstm80_loc]))

            v_mlp20_g = float(np.exp(m_mlp20_g(x20).item()))
            v_mlp80_g = float(np.exp(m_mlp80_g(x80).item()))

            rows.append({
                'shock_pct': int(s),
                'return': float(r),
                'baseline_variance': base_var,
                'baseline_logvol': baseline_logvol,
                'Base LSTM-20 Global': v_lstm20_g,
                'Base LSTM-80 Global': v_lstm80_g,
                'Base LSTM-80 Local': v_lstm80_l,
                'MLP-20 Global': v_mlp20_g,
                'MLP-80 Global': v_mlp80_g,
                'HARNet-20 Global': v_harnet20_g,
                'HARNet-80 Global': v_harnet80_g,
                'HARNet-80 Local': v_harnet80_l,
                'RFSV-20 Local': v_rfsv20,
                'RFSV-80 Local': v_rfsv80,
            })

    return pd.DataFrame(rows)


def plot_mcs_nic(df, output_path):
    """Plot News Impact Curves for the eight non-RFSV MCS models.

    Uses a clean, high-clarity layout showing eight models with consistent
    color palettes grouped by model family.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    # Styling per family
    # 1. Base LSTM family (blues)
    ax.plot(df['shock_pct'], df['Base LSTM-20 Global'], marker='o', lw=1.8,
            color='#1f77b4', label='Base LSTM-20 Global')
    ax.plot(df['shock_pct'], df['Base LSTM-80 Global'], marker='s', lw=1.8,
            color='#08519c', label='Base LSTM-80 Global')
    ax.plot(df['shock_pct'], df['Base LSTM-80 Local'], marker='^', lw=1.5, ls='--',
            color='#6baed6', label='Base LSTM-80 Local (5-ticker mean)')

    # 2. MLP family (reds/oranges)
    ax.plot(df['shock_pct'], df['MLP-20 Global'], marker='v', lw=1.8,
            color='#d95f02', label='MLP-20 Global')
    ax.plot(df['shock_pct'], df['MLP-80 Global'], marker='d', lw=1.8,
            color='#a63603', label='MLP-80 Global')

    # 3. HARNet family (greens)
    ax.plot(df['shock_pct'], df['HARNet-20 Global'], marker='<', lw=1.5,
            color='#2ca02c', label='HARNet-20 Global')
    ax.plot(df['shock_pct'], df['HARNet-80 Global'], marker='>', lw=1.5,
            color='#006d2c', label='HARNet-80 Global')
    ax.plot(df['shock_pct'], df['HARNet-80 Local'], marker='p', lw=1.3, ls='--',
            color='#74c476', label='HARNet-80 Local (5-ticker mean)')

    ax.set_xticks(np.arange(-5, 6, 1))
    ax.set_xlabel('Preceding Return Shock (%) [100 * log(Close / Open)]', fontsize=11)
    ax.set_ylabel('Forecast Daily Garman–Klass Variance', fontsize=11)
    ax.set_title('News Impact Curves at the Global Training Variance Floor', fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9.5)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('imgs/data_properties'))
    parser.add_argument('--checkpoints', type=Path, default=CHECKPOINT_ROOT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = load_mcs_models(args.checkpoints)
    df = simulate_mcs_nic(models)

    # Save CSV
    csv_path = args.output_dir / 'mcs_news_impact_curve.csv'
    df.to_csv(csv_path, index=False)

    # Save Plot
    png_path = args.output_dir / 'mcs_news_impact_curve.png'
    plot_mcs_nic(df, png_path)

    print(f'Saved MCS NIC table to {csv_path}')
    print(f'Saved MCS NIC plot to {png_path}')
    print('\nSimulated News Impact Curve values:')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
