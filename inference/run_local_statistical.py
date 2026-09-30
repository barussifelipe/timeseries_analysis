"""Fit five stocks independently with the saved global statistical specifications."""

import os
import subprocess
import sys
from pathlib import Path

from inference.run_local_base_lstm import DATABASE, ROOT, TICKERS


def main():
    jobs = []
    for kind in ('ar1', 'har', 'har_80', 'garch', 'sarima', 'rfsv'):
        for ticker in TICKERS:
            suffix = 'raw_history_full' if kind == 'rfsv' else f'raw_history_w{80 if kind == "har_80" else 20}'
            directory = ROOT / 'inference/checkpoints' / kind / 'garman-klass/local' / ticker / suffix
            fit, log = directory / 'fit.json', directory / 'fit.log'
            if any(path.exists() for path in (fit, log, directory / 'fit_failure.json')):
                raise FileExistsError(f'refusing to overwrite {directory}')
            jobs.append((kind, ticker, 80 if kind == 'har_80' else 20, directory, log))
    for kind, ticker, window, directory, log in jobs:
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-m', f'models.{kind}', '--database', DATABASE,
                   '--estimator', 'garman-klass', '--scope', 'local', '--ticker', ticker,
                   '--window-size', str(window), '--raw-fit-only']
        print(f'START {kind} {ticker}', flush=True)
        with log.open('x', encoding='utf-8') as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, check=False)
        print(f'END {kind} {ticker}: exit {result.returncode}; log {log}', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
