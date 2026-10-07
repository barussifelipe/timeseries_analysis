"""Calculate five-stock frictionless realized utility from saved MCS forecasts."""

import argparse
import csv
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


MODELS = (
    'Base LSTM-20 GLOBAL', 'Base LSTM-80 GLOBAL', 'Base LSTM-80 LOCAL',
    'HARNet-20 GLOBAL', 'HARNet-80 GLOBAL', 'HARNet-80 LOCAL',
    'MLP-20 GLOBAL', 'MLP-80 GLOBAL', 'RFSV-20 LOCAL', 'RFSV-80 LOCAL',
)
TICKERS = ('AAPL', 'AMZN', 'GOOG', 'NFLX', 'NVDA')


def realized_utility(actual, predicted, sharpe, gamma=2):
    ratio = actual / predicted
    return sharpe ** 2 / gamma * math.sqrt(ratio) - sharpe ** 2 / (2 * gamma) * ratio


def calculate(database, forecast_root):
    forecasts = {}
    actuals = {}
    for window in (20, 80):
        wanted = {name for name in MODELS if f'-{window} ' in name}
        with (forecast_root / f'{window}_size' / 'forecasts.csv').open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                model = row['candidate']
                if model not in wanted:
                    continue
                key = (row['ticker'], row['date'])
                if key[0] not in TICKERS or not '2019-01-01' <= key[1] < '2026-01-01':
                    raise ValueError(f'{model}: unexpected key {key}')
                values = forecasts.setdefault(model, {})
                if key in values:
                    raise ValueError(f'{model}: duplicate forecast key {key}')
                actual, predicted = float(row['actual']), float(row['predicted'])
                if not all(math.isfinite(v) and v > 0 for v in (actual, predicted)):
                    raise ValueError(f'{model}: invalid variance at {key}')
                if key in actuals and actuals[key] != actual:
                    raise ValueError(f'{model}: target disagrees at {key}')
                actuals[key] = actual
                values[key] = predicted

    if set(forecasts) != set(MODELS) or len(actuals) != 8800:
        raise ValueError('missing model or expected 8,800 common targets')
    keys = set(actuals)
    dates = {date for _, date in keys}
    if len(dates) != 1760 or any({date for ticker, date in keys if ticker == name} != dates for name in TICKERS):
        raise ValueError('expected 1,760 identical test dates per ticker')
    if any(set(values) != keys for values in forecasts.values()):
        raise ValueError('forecast ticker-date keys differ across models')

    prices = {}
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        for ticker in TICKERS:
            for date, opened, closed in conn.execute('''SELECT Date, Open, Close FROM raw_history
                    WHERE Ticker = ? AND Date >= '2019-01-01' AND Date < '2026-01-01' ''', (ticker,)):
                key = (ticker, date)
                if key in prices:
                    raise ValueError(f'duplicate adjusted price key {key}')
                if key in keys:
                    prices[key] = (opened, closed)
    if set(prices) != keys:
        raise ValueError(f'missing adjusted price keys: {len(keys - set(prices))}')
    returns = defaultdict(list)
    variances = defaultdict(list)
    for key in sorted(keys):
        opened, closed = prices[key]
        if not all(v is not None and math.isfinite(v) and v > 0 for v in (opened, closed)):
            raise ValueError(f'invalid adjusted Open/Close at {key}')
        ret = math.expm1(math.log(closed / opened))
        if not math.isfinite(ret):
            raise ValueError(f'invalid adjusted return at {key}')
        returns[key[0]].append(ret)
        variances[key[0]].append(actuals[key])
    mean_return = sum(sum(returns[t]) / 1760 for t in TICKERS) / 5
    mean_variance = sum(sum(variances[t]) / 1760 for t in TICKERS) / 5
    daily_rf = (1 + 0.0426) ** (1 / 252) - 1
    sharpe = (mean_return - daily_rf) / math.sqrt(mean_variance)
    scores = {}
    for model in MODELS:
        ticker_scores = {ticker: [] for ticker in TICKERS}
        for key in sorted(keys):
            score = realized_utility(actuals[key], forecasts[model][key], sharpe)
            if not math.isfinite(score):
                raise ValueError(f'{model}: invalid utility at {key}')
            ticker_scores[key[0]].append(score)
        scores[model] = 100 * sum(sum(ticker_scores[t]) / 1760 for t in TICKERS) / 5
    return scores, mean_return, mean_variance, daily_rf, sharpe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('D:/DBs/timeseries_analysis/history_coverage.db'))
    parser.add_argument('--forecasts', type=Path, default=Path('imgs/eval/mcs'))
    parser.add_argument('--output', type=Path, default=Path('imgs/eval/utility/realized_utility.csv'))
    args = parser.parse_args()
    scores, mean_return, mean_variance, daily_rf, sharpe = calculate(args.database, args.forecasts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('model', 'ru_daily_percent_of_wealth'))
        writer.writerows((model, f'{scores[model]:.12g}') for model in MODELS)
    print(f'matched keys/model=8800; dates/ticker=1760; mean return={mean_return:.12g}; '
          f'mean GK variance={mean_variance:.12g}; daily rf={daily_rf:.12g}; SR={sharpe:.12g}')
    for model in MODELS:
        print(f'{model}: {scores[model]:.12g}%')


if __name__ == '__main__':
    main()
