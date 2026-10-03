"""Sequential Garman-Klass validation sweep; reruns skip completed trials."""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

from models.training_blocks import load_fit


RESULTS = Path('ref/implementation_plan/lstm_val_training.md')
LOGS = Path('wandb/lstm_val_training/logs')
METRICS = ('qlike', 'mse', 'rmse', 'mase', 'mae')


def write_results(rows):
    rows = [row for row in rows if not row['name'].endswith('_preflight')]
    lines = [
        '# Garman–Klass LSTM validation training', '',
        'Daily unannualized reduced Garman–Klass variance from adjusted OHLC:',
        '0.5 ln(High/Low)² − (2 ln 2 − 1) ln(Close/Open)². Forecast the next',
        'observation from the previous',
        '20, 40, or 80 within-ticker values. Full 1,525-equity panel; pre-2016 fit,',
        '2016–2018 validation selection. The 2019–2025 test period is not evaluated.',
        'Seed 42, batch 128, QLIKE checkpoint selection, one learning-rate retry.',
        'Original sweep: up to 20 epochs and patience 5. Selected trials use',
        'patience 10 before and after one rate retry. Hidden-16 trials allow up to',
        '50 epochs; raw hidden-64 was extended from 10 to 20, and log-volatility',
        'hidden-64 and single-input hidden-128 trials allow up to 20 epochs;',
        'selected two-input hidden-128 runs allow up to 50 epochs.',
        '`_logvol_` runs use log(sqrt(variance)) inputs and next-day targets.',
        '_intradayret_ LSTM runs add ln(adjusted Close/Open) as the second input.',
        'Their QLIKE objective and reported errors reconstruct variance with exp(2*target).',
        'The original sweep uses gradient norm cap 1; `_noclip` runs disable clipping.',
        'Each metric below comes from the best validation-QLIKE epoch. Lower is better.',
        '',
        '| Run | Status | Epochs | Best epoch | QLIKE | MSE | RMSE | MASE | MAE | W&B | Log |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |',
    ]
    for row in rows:
        metrics = row.get('metrics', {})
        values = [f'{metrics[key]:.9g}' if key in metrics else '—' for key in METRICS]
        url = row.get('url', '')
        link = f'[run]({url})' if url else '—'
        log = row['log']
        lines.append(f"| {row['name']} | {row['status']} | {row.get('epochs', '—')} | "
                     f"{row.get('best_epoch', '—')} | {' | '.join(values)} | {link} | "
                     f"[{Path(log).name}](../../{log.replace(os.sep, '/')}) |")
    incomplete = [row for row in rows if row['status'] != 'complete']
    if incomplete:
        lines += ['', '## Incomplete or failed runs', '']
        lines += [f"- {row['name']}: {row.get('error', 'see log')}" for row in incomplete]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True)
    parser.add_argument('--preflight', action='store_true', help='one largest trial with limited rows and tickers')
    parser.add_argument('--no-clip-trial', action='store_true',
                        help='run only base LSTM, window 20, width 16, rate 0.0001 without clipping')
    parser.add_argument('--progress-batches', type=int, default=0,
                        help='print batch progress at this interval and metrics after each epoch')
    args = parser.parse_args()
    if args.progress_batches < 0:
        parser.error('progress interval must be nonnegative')
    database = Path(args.database).resolve()
    if not database.is_file():
        parser.error('database does not exist')
    stat = database.stat()
    database_id = [str(database), stat.st_size, stat.st_mtime_ns]
    import wandb
    viewer = wandb.Api(timeout=15).viewer
    if not viewer:
        raise RuntimeError('online W&B access unavailable')
    LOGS.mkdir(parents=True, exist_ok=True)
    state = LOGS.parent / 'results.json'
    rows = json.loads(state.read_text(encoding='utf-8')) if state.exists() else []
    if not args.preflight and any(row.get('database_id') != database_id for row in rows
                                  if not row['name'].endswith('_preflight')):
        parser.error('results already belong to another database version')
    existing = {row['name']: row for row in rows}
    combinations = ([('base_lstm_vol', 20, 16, 1e-4)] if args.no_clip_trial else
                    [('silu_lstm', 80, 64, 1e-3)] if args.preflight else
                    itertools.product(('base_lstm_vol', 'silu_lstm'), (20, 40, 80),
                                      (16, 32, 64), (1e-3, 1e-4)))
    selected = []
    for kind, window, hidden, rate in combinations:
        name = (f'{kind}_gk_w{window}_h{hidden}_lr{rate:g}'
                + ('_noclip' if args.no_clip_trial else '')
                + ('_progress' if args.progress_batches else '')
                + ('_preflight' if args.preflight else ''))
        selected.append(name)
        if (existing.get(name, {}).get('status') == 'complete'
                and existing[name].get('database_id') == database_id):
            continue
        log = LOGS / f'{name}.log'
        command = [sys.executable, '-m', 'models.base_lstm' if kind == 'base_lstm_vol' else 'models.silu_lstm',
                   '--database', str(database), '--estimator', 'garman-klass', '--scope', 'global',
                   '--run-name', name, '--window-size', str(window), '--hidden-size', str(hidden),
                   '--learning-rate', str(rate), '--batch-size', '128', '--epochs', '20', '--validation-only']
        if args.preflight:
            command += ['--limit-tickers', '2', '--limit-rows', '100', '--epochs', '1']
        if args.no_clip_trial:
            command += ['--no-grad-clip']
        if args.progress_batches:
            command += ['--log-every-batches', str(args.progress_batches)]
        with log.open('w', encoding='utf-8') as output:
            outcome = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                     env={**os.environ, 'WANDB_MODE': 'online'}, check=False)
        text = log.read_text(encoding='utf-8', errors='replace')
        messages = []
        for line in text.splitlines():
            try:
                messages.append(json.loads(line))
            except ValueError:
                pass
        url = next((message['wandb_url'] for message in messages if 'wandb_url' in message), '')
        device = next((message['training_device'] for message in messages if 'training_device' in message), '')
        path = Path('inference/checkpoints') / kind / 'garman-klass' / 'global' / name / 'fit.pth'
        row = {'name': name, 'log': str(log), 'url': url, 'database_id': database_id,
               'status': 'complete' if outcome.returncode == 0 else 'failed'}
        if row['status'] == 'complete':
            try:
                fit = load_fit(path)
                row.update(epochs=fit['completed_epochs'], best_epoch=fit['epoch'],
                           metrics=fit['best_metrics'], device=device)
                if not url or device != 'cuda':
                    row.update(status='failed', error='missing online W&B URL or GPU execution')
            except (OSError, KeyError, RuntimeError) as error:
                row.update(status='failed', error=f'checkpoint read failed: {error}')
        else:
            row['error'] = f'exit code {outcome.returncode}; see console log'
        existing[name] = row
        rows = list(existing.values())
        state.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
        if not args.preflight:
            write_results([item for item in rows if not item['name'].endswith('_preflight')])
        print(json.dumps(row), flush=True)
        if args.preflight and row['status'] != 'complete':
            raise RuntimeError('preflight failed')
    if not args.preflight and any(existing[name]['status'] != 'complete' for name in selected):
        raise RuntimeError('one or more requested trials failed; see results table')


if __name__ == '__main__':
    main()
