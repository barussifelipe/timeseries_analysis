"""Fit pooled statistical models on raw-history ML window cohorts."""

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

from models.raw_neural_data import load_raw_neural
from models.training_blocks import TimeSeriesDataset


EXPECTED = {20: {'train': 8664516, 'val': 1806548, 'test': 4479681},
            80: {'train': 7004011, 'val': 1692154, 'test': 4295773}}
LOCAL_COUNTS = {
    'NVDA': {20: (4223, 754, 1760), 80: (4103, 754, 1760)},
    'AAPL': {20: (8408, 754, 1760), 80: (7915, 754, 1760)},
    'NFLX': {20: (3407, 754, 1760), 80: (3347, 754, 1760)},
    'GOOG': {20: (2843, 754, 1760), 80: (2783, 754, 1760)},
    'AMZN': {20: (4669, 754, 1760), 80: (4609, 754, 1760)},
}


def eligible_data(database, limit_tickers=None, window=20, ticker=None):
    if window not in EXPECTED:
        raise ValueError('unsupported raw-history window')
    frame, _, floor = load_raw_neural(database, ticker=ticker, limit_tickers=limit_tickers)
    frame = frame.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    counts = {}
    train_indices = None
    for split in EXPECTED[window]:
        data = TimeSeriesDataset(frame, window, split=split, valid_column='Valid')
        counts[split] = len(data)
        if split == 'train':
            train_indices = data.target_indices.numpy().copy()
        del data
    expected = (dict(zip(('train', 'val', 'test'), LOCAL_COUNTS[ticker][window]))
                if ticker in LOCAL_COUNTS else EXPECTED[window] if ticker is None and limit_tickers is None else None)
    if expected is not None and counts != expected:
        raise ValueError(f'raw-history window counts differ from ML: {counts} != {expected}')
    return frame, floor, counts, train_indices


def training_segments(frame):
    """Positive-input runs; a valid zero target closes its run."""
    for _, group in frame[frame.Date < '2016-01-01'].groupby('Ticker', sort=False, observed=True):
        values = group.Variance.to_numpy(dtype=float)
        raw_values = group.RawVariance.to_numpy(dtype=float)
        returns = group.IntradayLogReturn.to_numpy(dtype=float)
        valid = group.Valid.to_numpy(dtype=bool)
        start = 0
        for i in range(len(group)):
            if not valid[i] or raw_values[i] <= 0:
                end = i + int(valid[i] and raw_values[i] == 0)
                if end - start >= 21:
                    yield values[start:end], returns[start:end]
                start = i + 1
        if len(group) - start >= 21:
            yield values[start:], returns[start:]


def streamed_ols(frame, targets, kind, scale):
    width = {'ar1': 2, 'har': 4, 'har_80': 6}[kind]
    xx = np.zeros((width, width))
    xy = np.zeros(width)
    values = frame.Variance.to_numpy(dtype=float)
    for indices in np.array_split(targets, max(1, (len(targets) + 49999) // 50000)):
        if not len(indices):
            continue
        features = [np.ones(len(indices)), values[indices - 1] / scale]
        if kind in ('har', 'har_80'):
            for width_lag in ((5, 20, 40, 80) if kind == 'har_80' else (5, 20)):
                features.append(sum(values[indices - lag] for lag in range(1, width_lag + 1)) / (width_lag * scale))
        x = np.column_stack(features)
        xx += x.T @ x
        xy += x.T @ (values[indices] / scale)
    scaled = np.linalg.lstsq(xx, xy, rcond=None)[0]
    scaled[0] *= scale
    return scaled, {'normal_matrix_condition': float(np.linalg.cond(xx))}


def sarima_objective(models, parameters):
    return -sum(model.loglike(parameters, transformed=False) for model in models)


def fit_sarima(segments):
    from models.sarima import SARIMA

    models = [SARIMA()._model(values) for values, _ in segments]
    if not models:
        raise ValueError('no eligible SARIMA segments')
    started = time.monotonic()
    print(json.dumps({'sarima_stage': 'seed_fit', 'segments': len(models)}), flush=True)
    seed = models[0].fit(disp=False, maxiter=200)
    initial = models[0].untransform_params(seed.params)
    print(json.dumps({'sarima_stage': 'pooled_fit', 'seed_converged': bool(seed.mle_retvals.get('converged')),
                      'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    evaluations = 0

    def objective(parameters):
        nonlocal evaluations
        value = sarima_objective(models, parameters)
        evaluations += 1
        print(json.dumps({'sarima_stage': 'pooled_evaluation', 'evaluation': evaluations,
                          'negative_log_likelihood': float(value),
                          'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
        return value

    result = minimize(objective, initial,
                      method='Powell', options={'maxiter': 100})
    parameters = models[0].transform_params(result.x)
    return parameters, {'converged': bool(result.success), 'message': str(result.message),
                        'iterations': int(result.nit), 'evaluations': evaluations,
                        'objective': float(result.fun)}


def garch_objective(residuals, parameters, scale):
    omega, alpha, beta = parameters
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e100
    total = 0.
    for residual in residuals:
        initial = max(float(np.mean(residual ** 2)), scale * 1e-8)
        driving = omega + alpha * np.r_[0., residual[:-1] ** 2]
        driving[0] = initial
        variance = lfilter([1.], [1., -beta], driving)
        # The first state is the segment's sample variance, as in garch_fit.
        total += float(np.sum(.5 * (np.log(variance) + residual ** 2 / variance)))
    return total


def fit_garch(segments):
    from models.garch import GARCH

    residuals = []
    for _, returns in segments:
        past_sum = np.r_[0., np.cumsum(returns[:-1])]
        residuals.append(returns - past_sum / np.maximum(np.arange(len(returns)), 1))
    if not residuals:
        raise ValueError('no eligible GARCH segments')
    scale = float(sum(np.sum(r ** 2) for r in residuals) / sum(map(len, residuals)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('GARCH residual scale must be positive')
    result = minimize(lambda p: garch_objective(residuals, p, scale),
                      [scale * .1, .1, .8], method='L-BFGS-B',
                      bounds=[(scale * 1e-8, None), (0, 1), (0, 1)],
                      options={'maxiter': 100})
    valid = bool(result.success and np.isfinite(result.x).all())
    try:
        GARCH(*result.x)
    except ValueError:
        valid = False
    return result.x, {'converged': valid, 'message': str(result.message),
                      'iterations': int(result.nit), 'objective': float(result.fun)}


def fit_rfsv_full(args):
    """Calibrate RFSV and check its forward pass on full train histories."""
    if args.window_size != 20:
        raise ValueError('--window-size is unused for full-history RFSV; omit it')
    if (args.estimator != 'garman-klass' or args.scope not in ('global', 'local')
            or (args.scope == 'local' and args.ticker not in LOCAL_COUNTS)
            or (args.scope == 'global' and args.ticker)
            or args.limit_rows or args.limit_tickers):
        raise ValueError('full-history RFSV fit requires Garman-Klass and an eligible scope/ticker without limits')
    from models.rfsv import RFSV
    from models.variance_fit import rfsv_fit

    path = Path('inference/checkpoints/rfsv/garman-klass') / args.scope
    if args.scope == 'local':
        path /= args.ticker
    path /= 'raw_history_full/fit.json'
    if path.exists() or path.with_name('fit_failure.json').exists():
        raise FileExistsError(f'refusing to overwrite existing fit or failure: {path}')
    started = time.monotonic()
    frame, _, floor = load_raw_neural(args.database, ticker=args.ticker)
    frame = frame.sort_values(['Ticker', 'Date'])
    cohort_tickers = int(frame.Ticker.nunique())
    if args.scope == 'local':
        from data.roughness_analysis import local_raw_gk_roughness
        parameters = local_raw_gk_roughness(frame, args.ticker)
    else:
        parameters = rfsv_fit('garman-klass', expected_cohort=f'raw_history_{cohort_tickers}',
                              forecast_window='full_positive_history')
    model = RFSV(parameters['H'], parameters['nu_squared'])
    train = frame[(frame.Date < '2016-01-01') & frame.Valid & (frame.RawVariance > 0)]
    checked = 0
    for _, group in train.groupby('Ticker', observed=True):
        predicted = model.forward(group.Variance.to_numpy(dtype=float))
        if not np.isfinite(predicted) or predicted <= 0:
            raise ValueError('RFSV full-history forward check returned invalid variance')
        checked += 1
    if not checked:
        raise ValueError('no valid pre-2016 RFSV histories')
    database = Path(args.database).resolve()
    artifact = {'model': 'rfsv', 'specification': 'RFSV Section 5, full positive observation history',
                'source': str(database), 'source_bytes': database.stat().st_size,
                'source_mtime_ns': database.stat().st_mtime_ns,
                'cohort_rule': '>80 pre-2016 raw rows and 2025-12-31 row',
                'train_end_exclusive': '2016-01-01',
                'target': 'adjusted daily unannualized Garman-Klass variance',
                'forecast_history_rule': 'all preceding valid positive observations within ticker',
                'forecast_horizon': 'next recorded observation',
                'scope': args.scope, 'ticker': args.ticker,
                'cohort_tickers': cohort_tickers, 'train_tickers_with_history': checked,
                'train_tickers_without_history': cohort_tickers - checked,
                'training_history_rows': len(train), 'floor': floor,
                'counts': eligible_data(args.database, window=20, ticker=args.ticker)[2] if args.scope == 'local' else None,
                'parameters': parameters,
                'diagnostics': {'converged': True, 'method': 'pre-2016 observation-lag roughness moments',
                                'forward_checked_training_tickers': checked},
                'elapsed_seconds': round(time.monotonic() - started, 1)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    reloaded = json.loads(path.read_text(encoding='utf-8'))
    if reloaded['parameters'] != parameters or reloaded['diagnostics']['forward_checked_training_tickers'] != checked:
        raise RuntimeError('saved full-history RFSV fit failed reload verification')
    print(json.dumps({'completed_fit': str(path), 'train_tickers_with_history': checked,
                      'training_history_rows': len(train)}), flush=True)
    return path


def fit_raw(kind, args):
    if kind not in ('ar1', 'har', 'har_80', 'sarima', 'garch', 'rfsv'):
        raise ValueError('unsupported raw statistical fit')
    if kind == 'rfsv':
        return fit_rfsv_full(args)
    window = 80 if kind == 'har_80' else 20
    if (args.estimator != 'garman-klass' or args.scope not in ('global', 'local')
            or (args.scope == 'local' and args.ticker not in LOCAL_COUNTS)
            or (args.scope == 'global' and args.ticker)
            or args.limit_rows or args.limit_tickers or args.window_size != window):
        raise ValueError(f'raw fit requires Garman-Klass, window {window}, and an eligible local ticker or global scope without limits')
    path = Path('inference/checkpoints') / kind / 'garman-klass' / args.scope
    if args.scope == 'local':
        path /= args.ticker
    path = path / f'raw_history_w{window}' / 'fit.json'
    if path.exists() or path.with_name('fit_failure.json').exists():
        raise FileExistsError(f'refusing to overwrite existing fit or failure: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frame, floor, counts, targets = eligible_data(args.database, window=window, ticker=args.ticker)
    segments = list(training_segments(frame))
    database = Path(args.database).resolve()
    metadata = {'model': kind, 'specification': {'ar1': 'AR(1)', 'har': 'HAR(1,5,20)',
                'har_80': 'HAR(1,5,20,40,80)',
                'sarima': 'SARIMA(1,0,1)x(1,0,1,5)', 'garch': 'GARCH(1,1)'}[kind],
                'scope': args.scope, 'ticker': args.ticker,
                'source': str(database), 'source_bytes': database.stat().st_size,
                'source_mtime_ns': database.stat().st_mtime_ns,
                'cohort_rule': '>80 pre-2016 rows and 2025-12-31 row',
                'target': 'adjusted daily unannualized Garman-Klass variance; zero targets floored',
                'garch_quantity': 'conditional variance of adjusted log(Close/Open) residuals' if kind == 'garch' else None,
                'train_end_exclusive': '2016-01-01', 'window': window, 'floor': floor,
                'cohort_tickers': int(frame.Ticker.nunique()), 'counts': counts,
                'valid_segments': len(segments),
                'segment_observations': sum(len(s[0]) for s in segments),
                'eligible_training_windows': counts['train'], 'invalid_source_rows': int((~frame.Valid).sum()),
                'zero_variance_rows': int((frame.Valid & frame.RawVariance.eq(0)).sum())}
    print(json.dumps({'pre_fit': metadata}), flush=True)
    try:
        if kind in ('ar1', 'har', 'har_80'):
            scale = float(frame.loc[frame.Date < '2016-01-01', 'Variance'].median())
            parameters, diagnostics = streamed_ols(frame, targets, kind, scale)
            diagnostics['converged'] = True
        elif kind == 'sarima':
            parameters, diagnostics = fit_sarima(segments)
        else:
            parameters, diagnostics = fit_garch(segments)
        if not diagnostics['converged'] or not np.isfinite(parameters).all():
            raise RuntimeError(f'{kind} did not converge: {diagnostics}')
        artifact = {**metadata, 'parameters': np.asarray(parameters).tolist(),
                    'diagnostics': diagnostics, 'elapsed_seconds': round(time.monotonic() - started, 1)}
        path.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
        reloaded = json.loads(path.read_text(encoding='utf-8'))
        if not np.isfinite(reloaded['parameters']).all() or not reloaded['diagnostics']['converged']:
            raise RuntimeError('saved fit failed reload verification')
        print(json.dumps({'completed_fit': str(path), 'diagnostics': diagnostics}), flush=True)
        return path
    except Exception as error:
        failure = path.with_name('fit_failure.json')
        failure.write_text(json.dumps({**metadata, 'error': str(error),
                                       'elapsed_seconds': round(time.monotonic() - started, 1)}, indent=2), encoding='utf-8')
        raise
