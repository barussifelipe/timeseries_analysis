"""Check the local Base LSTM queue without starting training."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from inference import run_local_base_lstm, run_local_harnet, run_local_mlp, run_local_silu_lstm


def test_local_base_lstm_jobs():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        calls = []

        def record(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0)

        with patch.object(run_local_base_lstm, 'ROOT', root), \
                patch.object(run_local_base_lstm, 'LOGS', root / 'logs'), \
                patch.object(run_local_base_lstm.subprocess, 'run', side_effect=record):
            run_local_base_lstm.main()
        assert len(calls) == 10
        assert {(cmd[cmd.index('--ticker') + 1], cmd[cmd.index('--window-size') + 1])
                for cmd in calls} == {(ticker, window) for ticker in run_local_base_lstm.TICKERS
                                     for window in ('20', '80')}
        assert all('--fit-only' in cmd and '--no-wandb' not in cmd for cmd in calls)


def test_local_harnet_jobs():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        calls = []

        def record(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0)

        with patch.object(run_local_harnet, 'ROOT', root), \
                patch.object(run_local_harnet, 'LOGS', root / 'logs'), \
                patch.object(run_local_harnet.subprocess, 'run', side_effect=record):
            run_local_harnet.main()
        assert len(calls) == 10
        assert {(cmd[2], cmd[cmd.index('--ticker') + 1], cmd[cmd.index('--window-size') + 1])
                for cmd in calls} == {(f'models.harnet_{window}', ticker, window)
                                     for ticker in run_local_harnet.TICKERS for window in ('20', '80')}
        assert all('--fit-only' in cmd and '--data-transform' in cmd
                   and cmd[cmd.index('--data-transform') + 1] == 'variance' for cmd in calls)


def test_local_mlp_jobs():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        calls = []

        def record(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0)

        with patch.object(run_local_mlp, 'ROOT', root), \
                patch.object(run_local_mlp, 'LOGS', root / 'logs'), \
                patch.object(run_local_mlp.subprocess, 'run', side_effect=record):
            run_local_mlp.main()
        assert len(calls) == 10
        assert {(cmd[cmd.index('--ticker') + 1], cmd[cmd.index('--window-size') + 1])
                for cmd in calls} == {(ticker, window) for ticker in run_local_mlp.TICKERS
                                     for window in ('20', '80')}
        assert all(cmd[2] == 'models.mlp' and '--fit-only' in cmd and
                   cmd[cmd.index('--data-transform') + 1] == 'log-volatility' and
                   cmd[cmd.index('--hidden-size') + 1] == '128' and
                   cmd[cmd.index('--training-loss') + 1] == 'qlike' for cmd in calls)


def test_local_silu_jobs_and_failure_stop():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        calls = []

        def record(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=0)

        with patch.object(run_local_silu_lstm, 'ROOT', root), \
                patch.object(run_local_silu_lstm, 'LOGS', root / 'logs'), \
                patch.object(run_local_silu_lstm.subprocess, 'run', side_effect=record):
            run_local_silu_lstm.main()
            try:
                run_local_silu_lstm.main()
            except FileExistsError:
                pass
            else:
                raise AssertionError('queue must refuse to overwrite logs')
        assert len(calls) == 10
        assert {(cmd[cmd.index('--ticker') + 1], cmd[cmd.index('--window-size') + 1])
                for cmd in calls} == {(ticker, window) for ticker in run_local_silu_lstm.TICKERS
                                     for window in ('20', '80')}
        assert all(cmd[2] == 'models.silu_lstm' and '--fit-only' in cmd and
                   cmd[cmd.index('--data-transform') + 1] == 'variance' and
                   cmd[cmd.index('--training-loss') + 1] == 'mse' and
                   cmd[cmd.index('--hidden-size') + 1] == '128' for cmd in calls)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.object(run_local_silu_lstm, 'ROOT', root), \
                patch.object(run_local_silu_lstm, 'LOGS', root / 'logs'), \
                patch.object(run_local_silu_lstm.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as failed:
            try:
                run_local_silu_lstm.main()
            except SystemExit as error:
                assert error.code == 1
            else:
                raise AssertionError('queue must stop on first failed fit')
            assert failed.call_count == 1
