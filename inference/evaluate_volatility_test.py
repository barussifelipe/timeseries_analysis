"""Score saved raw-history GK forecasts on each model's full eligible test set."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.support_scripts.raw_neural_data import load_raw_neural
from models.support_scripts.raw_statistical_fit import EXPECTED
from models.training_blocks import TimeSeriesDataset, load_fit
from models.variance_neural import make_model, model_data


ROOT = Path('inference/checkpoints')
NEURAL = {
    'MLP-20': ('mlp', 'mlp_raw_gk_logvol_return_w20_h128_lr0p001_20e_qlike_online', 20),
    'MLP-80': ('mlp', 'mlp_raw_gk_logvol_return_w80_h128_lr0p001_20e_qlike_online', 80),
    'Base LSTM-20': ('base_lstm_vol', 'raw_gk_logvol_return_w20_h128_lr0p001_30e', 20),
    'Base LSTM-80': ('base_lstm_vol', 'base_lstm_raw_gk_logvol_return_w80_h128_lr0p001_20e_wandb', 80),
    'SiLU-LSTM-20': ('silu_lstm', 'silu_lstm_raw_gk_variance_return_direct_mse_w20_h128_lr0p001_20e_wandb', 20),
    'SiLU-LSTM-80': ('silu_lstm', 'silu_lstm_raw_gk_variance_return_direct_mse_w80_h128_lr0p001_20e_wandb', 80),
    'HARNet-20': ('harnet_20', 'harnet_20_raw_gk_variance_w20_lr0p001_20e_qlike_online', 20),
    'HARNet-80': ('harnet_80', 'harnet_80_raw_gk_variance_w80_lr0p001_20e_qlike_online', 80),
}
STATISTICAL = {'AR(1)': ('ar1', 20), 'HAR-20': ('har', 20),
               'HAR-80': ('har_80', 80), 'GARCH(1,1)': ('garch', 20),
               'SARIMA': ('sarima', 20)}
RFSV_FULL = ROOT / 'rfsv/garman-klass/global/raw_history_full/fit.json'
RFSV_WINDOWS = {'RFSV-20': 20, 'RFSV-80': 80}


def artifacts(database, frame, floor):
    """Reject missing or mismatched sources before scoring any row."""
    source = Path(database).resolve()
    cohort = tuple(frame.Ticker.drop_duplicates())
    result = {}
    for label, (kind, run, window) in NEURAL.items():
        path = ROOT / kind / 'garman-klass/global' / run / 'fit.pth'
        fit = load_fit(path)
        settings = fit['settings']
        if (settings.get('model') != kind or settings.get('window_size') != window
                or settings.get('estimator') != 'garman-klass' or not settings.get('raw_history')
                or Path(settings['database']).resolve() != source
                or tuple(settings['cohort_tickers']) != cohort
                or settings.get('counts') != EXPECTED[window] or fit['floor'] != floor
                or fit.get('output_convention') != settings.get('output_convention')):
            raise ValueError(f'{label}: incompatible neural artifact')
        settings.pop('training_history', None)
        settings.pop('cohort_tickers', None)
        result[label] = (path, fit, window)
    for label, (kind, window) in STATISTICAL.items():
        path = ROOT / kind / 'garman-klass/global' / f'raw_history_w{window}' / 'fit.json'
        fit = json.loads(path.read_text(encoding='utf-8'))
        if (fit['model'] != kind or Path(fit['source']).resolve() != source
                or fit['source_bytes'] != source.stat().st_size
                or fit['source_mtime_ns'] != source.stat().st_mtime_ns
                or fit['cohort_tickers'] != len(cohort) or fit['counts'] != EXPECTED[window]
                or fit['window'] != window or fit['floor'] != floor
                or not fit['diagnostics']['converged']):
            raise ValueError(f'{label}: incompatible statistical artifact')
        result[label] = (path, fit, window)
    fit = json.loads(RFSV_FULL.read_text(encoding='utf-8'))
    if (fit['model'] != 'rfsv' or Path(fit['source']).resolve() != source
            or fit['source_bytes'] != source.stat().st_size
            or fit['source_mtime_ns'] != source.stat().st_mtime_ns
            or fit['cohort_tickers'] != len(cohort) or fit['floor'] != floor
            or fit['parameters']['forecast_window'] != 'full_positive_history'
            or fit['forecast_history_rule'] != 'all preceding valid positive observations within ticker'
            or not fit['diagnostics']['converged']):
        raise ValueError('RFSV-full: incompatible statistical artifact')
    for label, window in RFSV_WINDOWS.items():
        result[label] = (RFSV_FULL, fit, window)
    return result


def scales_from_test(frame, data):
    """Ticker naive MAE on exactly the target dates scored by this window."""
    targets = data.target_indices.numpy()
    ids = data.ticker_ids.numpy()
    values = frame.Variance.to_numpy(dtype=float)
    naive_error = np.abs(values[targets] - values[targets - 1])
    if not np.isfinite(naive_error).all():
        raise ValueError('test naive errors must be finite')
    size = len(data.ticker_names)
    sums = np.bincount(ids, weights=naive_error, minlength=size)
    counts = np.bincount(ids, minlength=size)
    scales = np.divide(sums, counts, out=np.full(size, np.nan), where=counts > 0)
    scales[scales <= 0] = np.nan
    return dict(zip(data.ticker_names, scales))


class Totals:
    def __init__(self):
        self.n = self.hits = 0
        self.mae = self.mase = self.mse = self.qlike = 0.

    def add(self, actual, raw, scales, floor):
        actual, raw, scales = (np.asarray(x, dtype=np.float64).reshape(-1)
                               for x in (actual, raw, scales))
        if (not len(actual) or not np.isfinite(actual).all() or (actual <= 0).any()
                or not np.isfinite(raw).all() or not np.isfinite(scales).all()
                or (scales <= 0).any()):
            raise ValueError('invalid forecast, actual, or MASE scale')
        predicted = np.maximum(raw, floor)
        log_ratio = np.log(actual) - np.log(predicted)
        error = np.abs(actual - predicted)
        self.n += len(actual)
        self.hits += int((raw < floor).sum())
        self.mae += float(error.sum())
        self.mase += float((error / scales).sum())
        self.mse += float(np.square(error).sum())
        self.qlike += float(np.expm1(log_ratio).sum() - log_ratio.sum())

    def result(self):
        if not self.n:
            raise ValueError('empty test row')
        values = {'N': self.n, 'floor_hit_pct': 100 * self.hits / self.n,
                  'MAE': self.mae / self.n, 'MASE': self.mase / self.n,
                  'MSE': self.mse / self.n, 'RMSE': np.sqrt(self.mse / self.n),
                  'QLIKE': self.qlike / self.n}
        if not np.isfinite(list(values.values())).all():
            raise ValueError('nonfinite test metric')
        return values


def score_neural(kind, fit, frame, data, scale_by_ticker, floor, batch_size, excluded_tickers=(), emit=None):
    settings = fit['settings']
    transformed, column = model_data(frame, settings['data_transform'])
    features = tuple(settings['feature_columns'])
    if features not in ((column,), (column, 'IntradayLogReturn')):
        raise ValueError('unsupported neural feature transform')
    values = torch.tensor(transformed[list(features)].fillna(0).to_numpy(dtype=np.float32))
    model = make_model(kind, settings['window_size'], settings['hidden_width'], len(features))
    model.load_state_dict(fit['model_state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    starts = data.valid_indices.numpy()
    targets = data.target_indices.numpy()
    ids = data.ticker_ids.numpy()
    names = data.ticker_names
    scales = np.array([scale_by_ticker.get(name, np.nan) for name in names])
    selected = np.flatnonzero(np.isfinite(scales[ids]) & (scales[ids] > 0)
                              & ~np.isin(np.asarray(names)[ids], list(excluded_tickers)))
    actual = frame.Variance.to_numpy(dtype=float)
    offsets = torch.arange(settings['window_size'])
    totals = Totals()
    with torch.inference_mode():
        for begin in range(0, len(selected), batch_size):
            batch = selected[begin:begin + batch_size]
            x = values[torch.as_tensor(starts[batch])[:, None] + offsets].to(device)
            output = model(x).double().reshape(-1)
            convention = fit['output_convention']
            raw = output.exp() if convention == 'log_variance' else output
            if convention not in ('log_variance', 'raw_variance', 'floored_variance'):
                raise ValueError('unsupported neural output convention')
            raw = raw.cpu().numpy()
            totals.add(actual[targets[batch]], raw, scales[ids[batch]], floor)
            if emit is not None:
                emit(frame.Date.iloc[targets[batch]].to_numpy(), actual[targets[batch]], np.maximum(raw, floor))
    return totals.result()


def statistical_predictions(kind, fit, group, target_positions, window=None):
    """Each prediction uses only earlier rows from this ticker."""
    p = fit['parameters']
    variance = group.Variance.to_numpy(dtype=float)
    valid = group.Valid.to_numpy(dtype=bool)
    n = len(group)
    out = np.full(len(target_positions), np.nan)
    if kind == 'ar1':
        return p[0] + p[1] * variance[target_positions - 1]
    if kind in ('har', 'har_80'):
        widths = (1, 5, 20, 40, 80) if kind == 'har_80' else (1, 5, 20)
        cumulative = np.r_[0., np.cumsum(np.where(valid, variance, 0.))]
        return p[0] + sum(p[j + 1] * (cumulative[target_positions] - cumulative[target_positions - width]) / width
                          for j, width in enumerate(widths))
    if kind == 'rfsv':
        from scipy.special import betainc
        from math import gamma
        h, nu = p['H'], p['nu_squared']
        if window not in (20, 80):
            raise ValueError('RFSV requires a 20- or 80-observation window')
        edges = np.arange(window + 1, dtype=float)
        mass = betainc(.5 - h, .5 + h, edges / (1 + edges))
        weights = np.diff(mass)
        weights[-1] += 1 - mass[-1]
        constant = 2 * gamma(1.5 - h) / (gamma(h + .5) * gamma(2 - 2 * h)) * nu
        for begin in range(0, len(out), 4096):
            pos = target_positions[begin:begin + 4096]
            out[begin:begin + len(pos)] = np.exp(np.log(variance[pos[:, None] - np.arange(1, window + 1)]) @ weights + constant)
        return out
    if kind == 'sarima':
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        observed = np.where(valid, variance, np.nan)
        model = SARIMAX(observed, order=(1, 0, 1), seasonal_order=(1, 0, 1, 5))
        return np.asarray(model.filter(p).get_prediction(start=0, end=n - 1).predicted_mean)[target_positions]
    if kind == 'garch':
        from models.garch import GARCH
        model = GARCH(*p)
        returns = group.IntradayLogReturn.to_numpy(dtype=float)
        raw_variance = group.RawVariance.to_numpy(dtype=float)
        state = model.omega / (1 - model.alpha - model.beta)
        prediction = np.full(n, np.nan)
        count = mean = 0.
        for i in range(n):
            prediction[i] = state
            if valid[i] and raw_variance[i] > 0:
                residual = returns[i] - mean
                state = model.omega + model.alpha * residual**2 + model.beta * state
                count += 1
                mean += (returns[i] - mean) / count
            else:
                state = model.omega / (1 - model.alpha - model.beta)
                count = mean = 0.
        return prediction[target_positions]
    raise ValueError(kind)


def score_statistical(kind, fit, frame, data, scale_by_ticker, floor, excluded_tickers=(), emit=None):
    targets = data.target_indices.numpy()
    actual = frame.Variance.to_numpy(dtype=float)
    totals = Totals()
    offset = 0
    for ticker, group in frame.groupby('Ticker', sort=False, observed=True):
        stop = offset + len(group)
        chosen = targets[np.searchsorted(targets, offset):np.searchsorted(targets, stop)]
        scale = scale_by_ticker.get(ticker, np.nan)
        if len(chosen) and ticker not in excluded_tickers and np.isfinite(scale) and scale > 0:
            raw = statistical_predictions(kind, fit, group, chosen - offset, data.window_size)
            totals.add(actual[chosen], raw, np.full(len(chosen), scale), floor)
            if emit is not None:
                emit(frame.Date.iloc[chosen].to_numpy(), actual[chosen], np.maximum(raw, floor))
        offset = stop
    return totals.result()


def save_table(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = ('MAE', 'MASE', 'MSE', 'RMSE', 'QLIKE')
    columns = ['model', 'N', *metrics, 'floor_hit_pct', 'artifact']
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.axis('off')
    displayed = [[r['model'], *(f"{float(r[k]):.6g}" for k in metrics),
                  f"{float(r['floor_hit_pct']):.3f}%"] for r in rows]
    table = ax.table(cellText=displayed, colLabels=['Model', *metrics, 'Floor hit'],
                     loc='center', cellLoc='right')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for col, metric in enumerate(metrics, start=1):
        ranked = sorted(range(len(rows)), key=lambda i: float(rows[i][metric]))
        for rank, color in enumerate(('#b00020', '#0057b8')[:len(ranked)]):
            cell = table[ranked[rank] + 1, col]
            cell.get_text().set_color(color)
            cell.get_text().set_weight('bold')
    ax.set_title('One-step adjusted Garman-Klass variance forecasts | 2019-01-01 to 2025-12-31', pad=20)
    fig.text(.5, .048, 'Red: lowest loss. Blue: second lowest. Daily, unannualized variance; MASE uses naive MAE on each row\'s scored test dates.',
             ha='center', fontsize=9)
    fig.text(.5, .023, 'RFSV uses the same eligible test targets and ticker/window MASE scales as the other rows at its window.',
             ha='center', fontsize=8)
    fig.savefig(output, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', default='D:/DBs/timeseries_analysis/history_coverage.db')
    parser.add_argument('--output', type=Path, default=Path('imgs/eval/volatility_test_metrics.png'))
    parser.add_argument('--batch-size', type=int, default=2048)
    args = parser.parse_args()
    frame, _, floor = load_raw_neural(args.database)
    frame = frame.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    fits = artifacts(args.database, frame, floor)
    print(json.dumps({'preflight': '15 artifacts verified', 'floor': floor}), flush=True)
    rows = []
    for window in (20, 80):
        data = TimeSeriesDataset(frame, window, split='test', valid_column='Valid')
        if len(data) != EXPECTED[window]['test']:
            raise ValueError(f'window {window}: unexpected test count {len(data)}')
        print(json.dumps({'window': window, 'test_count': len(data)}), flush=True)
        scale_by_ticker = scales_from_test(frame, data)
        names = data.ticker_names
        eligible = np.array([np.isfinite(scale_by_ticker.get(name, np.nan)) and scale_by_ticker.get(name, np.nan) > 0
                             for name in names])
        scored_count = int(eligible[data.ticker_ids.numpy()].sum())
        excluded = len(data) - scored_count
        print(json.dumps({'window': window, 'scored_count': scored_count,
                          'excluded_no_test_scale': excluded}), flush=True)
        for label, (path, fit, fitted_window) in fits.items():
            if fitted_window != window:
                continue
            try:
                result = (score_neural(NEURAL[label][0], fit, frame, data, scale_by_ticker, floor, args.batch_size)
                          if label in NEURAL else
                          score_statistical('rfsv' if label in RFSV_WINDOWS else STATISTICAL[label][0],
                                            fit, frame, data, scale_by_ticker, floor))
                if result['N'] != scored_count:
                    raise ValueError('scored count differs from common MASE-eligible test count')
                row = {'model': label, **result, 'excluded_no_test_scale': excluded, 'artifact': str(path)}
                rows.append(row)
                print(json.dumps(row), flush=True)
            except Exception as error:
                print(json.dumps({'failed_row': label, 'error': str(error)}), flush=True)
                raise
        del data
    save_table(rows, args.output)
    print(json.dumps({'completed_rows': len(rows), 'png': str(args.output),
                      'csv': str(args.output.with_suffix('.csv'))}), flush=True)


if __name__ == '__main__':
    main()
