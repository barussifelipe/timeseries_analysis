"""The intraday queue stops after a failed trial."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_failed_trial_does_not_start_following_runs(tmp_path):
    source = Path('wandb/lstm_val_training/run_intraday_queue.py')
    spec = importlib.util.spec_from_file_location('intraday_queue', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    database = tmp_path / 'data.db'
    database.write_bytes(b'fixture')
    log_dir = tmp_path / 'logs'
    log_dir.mkdir()
    state = tmp_path / 'results.json'
    state.write_text(json.dumps([{
        'name': 'reference', 'status': 'complete',
        'database_id': [str(database), database.stat().st_size, database.stat().st_mtime_ns],
    }]), encoding='utf-8')
    def fail_first(command, **kwargs):
        kwargs['stdout'].write('{"progress": "run_end", "status": "failed"}\n')
        return SimpleNamespace(returncode=1)
    with patch.object(module, 'DATABASE', database), patch.object(module, 'LOGS', log_dir), \
         patch.object(module, 'STATE', state), patch.object(module, 'write_results'), \
         patch.object(module.subprocess, 'run', side_effect=fail_first) as run:
        try:
            module.main()
        except RuntimeError as error:
            assert 'later trials remain queued' in str(error)
        else:
            raise AssertionError('failed first trial did not stop queue')
    assert run.call_count == 1
    rows = json.loads(state.read_text(encoding='utf-8'))
    assert [row['status'] for row in rows[1:]] == ['failed', 'queued', 'queued']
    assert '--intraday-return' in run.call_args.args[0]
    assert run.call_args.args[0][run.call_args.args[0].index('--hidden-size') + 1] == '128'
