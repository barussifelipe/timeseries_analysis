"""Reassess both MCS block lengths on all saved 20/80-window forecasts."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from models.mcs_definition import model_confidence_set, qlike_losses


ROOT = Path('imgs/eval/mcs')


def load_forecasts(path):
    by_stock = defaultdict(dict)
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            by_stock[row['ticker']].setdefault(row['candidate'], []).append(row)
    return by_stock


def aligned_losses(sources, ticker):
    candidates = {}
    predictions = {}
    dates_ref = actual_ref = None
    for source in sources:
        for name, rows in source[ticker].items():
            dates = [row['date'] for row in rows]
            actual = np.array([float(row['actual']) for row in rows])
            predicted = np.array([float(row['predicted']) for row in rows])
            if len(dates) != 1760 or dates != sorted(set(dates)):
                raise ValueError(f'{ticker} {name}: incomplete or duplicate dates')
            if dates_ref is None:
                dates_ref, actual_ref = dates, actual
            elif dates != dates_ref or not np.array_equal(actual, actual_ref):
                raise ValueError(f'{ticker} {name}: dates or actuals differ')
            losses = qlike_losses(actual, predicted)
            saved = np.array([float(row['qlike']) for row in rows])
            if not np.allclose(losses, saved, rtol=1e-12, atol=1e-12):
                raise ValueError(f'{ticker} {name}: saved QLIKE differs')
            if name in candidates:
                if not np.array_equal(predicted, predictions[name]):
                    raise ValueError(f'{ticker}: shared candidate {name} has different forecasts')
                continue
            candidates[name] = losses
            predictions[name] = predicted
    names = sorted(candidates)
    if len(names) != 30 or len(dates_ref) != 1760:
        raise ValueError(f'{ticker}: expected 30 candidates and 1760 observations')
    return names, np.column_stack([candidates[name] for name in names]), dates_ref


def main():
    paths = [ROOT / f'{window}_size' / 'forecasts.csv' for window in (20, 80)]
    sources = [load_forecasts(path) for path in paths]
    tickers = sorted(sources[0])
    if len(tickers) != 5 or any(sorted(source) != tickers for source in sources):
        raise ValueError('expected the same five stocks in both source files')
    aligned = {ticker: aligned_losses(sources, ticker) for ticker in tickers}
    if any(aligned[ticker][0] != aligned[tickers[0]][0] for ticker in tickers):
        raise ValueError('candidate identifiers differ across stocks')
    seeds = np.random.SeedSequence(42).spawn(len(tickers))
    provenance = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in paths]
    destination = ROOT / 'all_windows'
    destination.mkdir(parents=True, exist_ok=True)
    for block_length in (20, 80):
        stocks = {}
        for ticker, child in zip(tickers, seeds):
            names, losses, dates = aligned[ticker]
            seed = int(child.generate_state(1)[0])
            result = model_confidence_set(losses, names, block_length=block_length, seed=seed)
            stocks[ticker] = {'n': len(dates), 'first_date': dates[0],
                              'last_date': dates[-1], 'seed': seed, **result}
        inclusion = {name: 100 * sum(name in stocks[ticker]['members'] for ticker in tickers)
                     / len(tickers) for name in aligned[tickers[0]][0]}
        report = {'metadata': {'sources': provenance, 'target': 'next recorded adjusted daily '
                              'unannualized GK variance', 'loss': 'dimensionless QLIKE',
                              'candidate_order': aligned[tickers[0]][0],
                              'block_length': block_length, 'repetitions': 10000,
                              'alpha': 0.10, 'seed': 42,
                              'observations_per_stock': 1760},
                  'stocks': stocks, 'inclusion_pct': inclusion}
        (destination / f'mcs_{block_length}.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8')
        shared = sorted(set.intersection(*(set(stock['members']) for stock in stocks.values())))
        lines = ['# Final models by stock', '',
                 f'All 30 candidates; {block_length}-observation bootstrap blocks; '
                 '1,760 shared targets per stock.', '']
        for ticker in tickers:
            lines.extend([f'## {ticker}', ''])
            lines.extend(f'- {name}' for name in stocks[ticker]['members'])
            lines.append('')
        lines.extend(['## Models in all five final sets', ''])
        lines.extend(f'- {name}' for name in shared)
        if not shared:
            lines.append('No model survives in all five stocks.')
        (destination / f'final_sets_{block_length}.md').write_text(
            '\n'.join(lines) + '\n', encoding='utf-8')
        print(f'MCS-{block_length}: {len(shared)} models in all five final sets')


if __name__ == '__main__':
    main()
