"""Continue the base raw-history run to 20 epochs, then start SiLU."""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
logs = root / 'wandb/lstm_val_training/logs'
base_name = 'raw_gk_logvol_return_w20_h128_lr0p001_30e'  # original checkpoint identity
silu_name = 'raw_gk_logvol_return_w20_h128_lr0p001_20e'
original_log = logs / f'base_lstm_vol_{base_name}.log'
base_checkpoint = root / 'inference/checkpoints/base_lstm_vol/garman-klass/global' / base_name / 'fit.pth'
common = ['--database', 'D:/DBs/timeseries_analysis/history_coverage.db',
          '--estimator', 'garman-klass', '--scope', 'global',
          '--raw-history', '--data-transform', 'log-volatility', '--intraday-return',
          '--window-size', '20', '--hidden-size', '128', '--learning-rate', '0.001',
          '--batch-size', '128', '--epochs', '20', '--patience', '10', '--compile',
          '--log-every-batches', '1000', '--validation-only', '--no-wandb']
jobs = [
    ('models.base_lstm', 'base_lstm_vol', base_name,
     logs / f'base_lstm_vol_{base_name}_resume_to20.log', base_checkpoint,
     ['--resume', '--resume-log', str(original_log)]),
    ('models.silu_lstm', 'silu_lstm', silu_name,
     logs / f'silu_lstm_{silu_name}.log',
     root / 'inference/checkpoints/silu_lstm/garman-klass/global' / silu_name / 'fit.pth', []),
]
if not original_log.exists() or not base_checkpoint.exists():
    raise FileNotFoundError('base run log and best checkpoint are required for continuation')
for _, _, _, log, checkpoint, extra in jobs:
    if log.exists() or (not extra and checkpoint.exists()):
        raise FileExistsError(f'refusing to overwrite {log} or {checkpoint}')
for module, kind, name, log, checkpoint, extra in jobs:
    command = [sys.executable, '-u', '-m', module, '--run-name', name, *common, *extra]
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
