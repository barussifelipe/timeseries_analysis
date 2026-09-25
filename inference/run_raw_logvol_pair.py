"""Run the two requested raw-history log-volatility trials sequentially."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
logs = root / 'wandb/lstm_val_training/logs'
logs.mkdir(parents=True, exist_ok=True)
name = 'raw_gk_logvol_return_w20_h128_lr0p001_30e'
common = ['--database', 'D:/DBs/timeseries_analysis/history_coverage.db',
          '--estimator', 'garman-klass', '--scope', 'global', '--run-name', name,
          '--raw-history', '--data-transform', 'log-volatility', '--intraday-return',
          '--window-size', '20', '--hidden-size', '128', '--learning-rate', '0.001',
          '--batch-size', '128', '--epochs', '30', '--patience', '10', '--compile',
          '--log-every-batches', '1000', '--validation-only', '--no-wandb']
jobs = [(module, kind, logs / f'{kind}_{name}.log',
         root / 'inference/checkpoints' / kind / 'garman-klass/global' / name / 'fit.pth')
        for module, kind in [('models.base_lstm', 'base_lstm_vol'),
                             ('models.silu_lstm', 'silu_lstm')]]
for _, _, log, checkpoint in jobs:
    if log.exists() or checkpoint.exists():
        raise FileExistsError(f'refusing to overwrite {log} or {checkpoint}')
for module, kind, log, checkpoint in jobs:
    command = [sys.executable, '-u', '-m', module, *common]
    print(json.dumps({'model': kind, 'status': 'starting', 'command': command,
                      'log': str(log), 'checkpoint': str(checkpoint)}), flush=True)
    started = time.monotonic()
    with log.open('x', encoding='utf-8') as output:
        result = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
    print(json.dumps({'model': kind, 'status': 'complete' if result.returncode == 0 else 'failed',
                      'returncode': result.returncode, 'elapsed_sec': round(time.monotonic()-started, 1),
                      'log': str(log), 'checkpoint': str(checkpoint)}), flush=True)
    if result.returncode:
        sys.exit(result.returncode)
