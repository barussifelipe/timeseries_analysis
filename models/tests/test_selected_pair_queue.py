"""A failed base process cannot launch the selected SiLU fit."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_failed_base_does_not_start_silu():
    path = Path('wandb/lstm_val_training/finish_selected_pair.py')
    spec = importlib.util.spec_from_file_location('selected_pair_queue', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = MagicMock()
    kernel.OpenProcess.return_value = 1
    kernel.WaitForSingleObject.return_value = 0
    def exited_badly(_handle, code):
        code._obj.value = 1
        return 1
    kernel.GetExitCodeProcess.side_effect = exited_badly
    with patch.object(module.ctypes, 'windll', SimpleNamespace(kernel32=kernel)), \
         patch.object(module, 'record') as record, \
         patch.object(module.subprocess, 'run') as run:
        try:
            module.main()
        except RuntimeError as error:
            assert 'exited unsuccessfully' in str(error)
        else:
            raise AssertionError('failed base process started SiLU')
        record.assert_not_called()
        run.assert_not_called()


if __name__ == '__main__':
    test_failed_base_does_not_start_silu()
