"""Fit the requested five-ticker direct-variance SiLU-LSTM group."""

import os
import subprocess
import sys

from inference.run_local_base_lstm import DATABASE, LOGS, ROOT, TICKERS


def main():
    jobs = []
    for ticker in TICKERS:
        for window in (20, 80):
            name = f'silu_lstm_local_{ticker}_raw_gk_variance_return_direct_mse_w{window}_h128_lr0p001_20e'
            checkpoint = ROOT / 'inference/checkpoints/silu_lstm/garman-klass/local' / ticker / name / 'fit.pth'
            log = LOGS / f'{name}.log'
            if checkpoint.exists() or log.exists():
                raise FileExistsError(f'refusing to overwrite {checkpoint} or {log}')
            jobs.append((ticker, window, name, log))
    LOGS.mkdir(parents=True, exist_ok=True)
    for ticker, window, name, log in jobs:
        command = [sys.executable, '-m', 'models.silu_lstm', '--database', DATABASE,
                   '--estimator', 'garman-klass', '--scope', 'local', '--ticker', ticker,
                   '--run-name', name, '--window-size', str(window), '--raw-history',
                   '--data-transform', 'variance', '--intraday-return',
                   '--training-loss', 'mse', '--hidden-size', '128', '--patience', '10',
                   '--epochs', '20', '--learning-rate', '0.001', '--batch-size', '128',
                   '--compile', '--log-every-batches', '1000', '--fit-only']
        print(f'START {ticker} w{window}: {name}', flush=True)
        with log.open('x', encoding='utf-8') as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle,
                                    stderr=subprocess.STDOUT,
                                    env={**os.environ, 'WANDB_MODE': 'online',
                                         'PYTHONDONTWRITEBYTECODE': '1'}, check=False)
        print(f'END {ticker} w{window}: exit {result.returncode}; log {log}', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
