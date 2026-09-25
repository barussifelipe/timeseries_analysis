"""Resume base at epoch 3 with backfilled W&B history, then fit SiLU online."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import wandb

root = Path(__file__).resolve().parents[1]
logs = root / 'wandb/lstm_val_training/logs'
base_name = 'raw_gk_logvol_return_w20_h128_lr0p001_30e'  # existing checkpoint identity
silu_name = 'silu_lstm_raw_gk_logvol_return_w20_h128_lr0p001_20e_wandb'
original_log = logs / f'base_lstm_vol_{base_name}.log'
continuation_log = logs / f'base_lstm_vol_{base_name}_resume_to20.log'
merged_log = logs / f'base_lstm_vol_{base_name}_epochs1_2_resume.jsonl'
base_log = logs / f'base_lstm_vol_{base_name}_resume_epoch3_wandb.log'
silu_log = logs / f'{silu_name}.log'
state_path = logs / 'raw_gk_logvol_pair_wandb_state.json'
base_checkpoint = root / 'inference/checkpoints/base_lstm_vol/garman-klass/global' / base_name / 'fit.pth'
silu_checkpoint = root / 'inference/checkpoints/silu_lstm/garman-klass/global' / silu_name / 'fit.pth'
if any(path.exists() for path in (merged_log, base_log, silu_log, state_path, silu_checkpoint)):
    raise FileExistsError('W&B continuation artifacts already exist')
if not all(path.exists() for path in (original_log, continuation_log, base_checkpoint)):
    raise FileNotFoundError('local epoch logs and epoch-2 checkpoint are required')
rows = [json.loads(line) for path in (original_log, continuation_log)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.startswith('{"progress": "epoch"')]
if [row['epoch'] for row in rows] != [1, 2]:
    raise ValueError('expected exactly completed epochs 1 and 2 in local logs')
fit = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
if fit['epoch'] != 2 or fit['settings']['run_name'] != base_name:
    raise ValueError('base checkpoint is not the expected epoch-2 run')
del fit
env = os.environ.copy()
env['WANDB_MODE'] = 'online'
os.environ['WANDB_MODE'] = 'online'
if not wandb.Api(timeout=15).viewer:
    raise RuntimeError('online W&B access unavailable')
run = wandb.init(project='timeseries-volatility', name=base_name,
                 config={'model': 'base_lstm_vol', 'epochs': 20, 'hidden_size': 128,
                         'learning_rate': 0.001, 'window_size': 20, 'batch_size': 128,
                         'patience': 10, 'raw_history': True,
                         'data_transform': 'log-volatility',
                         'feature_columns': ['LogVolatility', 'IntradayLogReturn'],
                         'resumed_from_local_best_epoch': 2,
                         'historical_metrics_backfilled_from': [str(original_log), str(continuation_log)]})
for row in rows:
    run.log({**{key: value for key, value in row.items() if key != 'progress'},
             'backfill/from_local_log': True})
state = {'base_wandb_id': run.id, 'base_wandb_url': run.url,
         'backfilled_epochs': [1, 2], 'source_logs': [str(original_log), str(continuation_log)]}
run.finish()
state_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
merged_log.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
print(json.dumps({'base_wandb_id': state['base_wandb_id'],
                  'base_wandb_url': state['base_wandb_url'],
                  'backfilled_epochs': [1, 2]}), flush=True)
common = ['--database', 'D:/DBs/timeseries_analysis/history_coverage.db',
          '--estimator', 'garman-klass', '--scope', 'global',
          '--raw-history', '--data-transform', 'log-volatility', '--intraday-return',
          '--window-size', '20', '--hidden-size', '128', '--learning-rate', '0.001',
          '--batch-size', '128', '--epochs', '20', '--patience', '10', '--compile',
          '--log-every-batches', '1000', '--validation-only']
jobs = [
    ('models.base_lstm', 'base_lstm_vol', base_name, base_log, base_checkpoint,
     ['--resume', '--resume-log', str(merged_log), '--wandb-id', state['base_wandb_id']]),
    ('models.silu_lstm', 'silu_lstm', silu_name, silu_log, silu_checkpoint, []),
]
for module, kind, name, log, checkpoint, extra in jobs:
    command = [sys.executable, '-u', '-m', module, '--run-name', name, *common, *extra]
    print(json.dumps({'model': kind, 'status': 'starting', 'command': command,
                      'log': str(log), 'checkpoint': str(checkpoint)}), flush=True)
    started = time.monotonic()
    with log.open('x', encoding='utf-8') as output:
        result = subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT, env=env)
    print(json.dumps({'model': kind, 'status': 'complete' if result.returncode == 0 else 'failed',
                      'returncode': result.returncode, 'elapsed_sec': round(time.monotonic()-started, 1),
                      'log': str(log), 'checkpoint': str(checkpoint)}), flush=True)
    if result.returncode:
        sys.exit(result.returncode)
