"""Final elapsed-time marker on completion and handled interruption."""

import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

from models.variance_neural import main

ARGS = ['base_lstm', '--database', 'unused', '--estimator', 'garman-klass',
        '--scope', 'global', '--validation-only']


def test_elapsed_marker():
    output = io.StringIO()
    with patch('sys.argv', ARGS), patch('models.variance_neural.neural_train', return_value='fit.pth'), \
         patch('models.variance_neural.load_fit', return_value={'settings': {'training_history': {}}}), \
         patch('models.variance_neural.prepare', return_value=(None, None, None)), \
         patch('models.variance_neural.neural_predict', return_value=[]), redirect_stdout(output):
        main('base_lstm_vol')
    end = json.loads(output.getvalue().splitlines()[-1])
    assert end['progress'] == 'run_end' and end['status'] == 'complete'
    assert end['elapsed_sec'] >= 0

    output = io.StringIO()
    with patch('sys.argv', ARGS), \
         patch('models.variance_neural.neural_train', side_effect=KeyboardInterrupt), \
         redirect_stdout(output):
        try:
            main('base_lstm_vol')
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('interrupt was swallowed')
    end = json.loads(output.getvalue().splitlines()[-1])
    assert end['progress'] == 'run_end' and end['status'] == 'interrupted'
    assert end['elapsed_sec'] >= 0

    output = io.StringIO()
    with patch('sys.argv', ARGS), \
         patch('models.variance_neural.neural_train', side_effect=ValueError('failed')), \
         redirect_stdout(output):
        try:
            main('base_lstm_vol')
        except ValueError:
            pass
        else:
            raise AssertionError('failure was swallowed')
    end = json.loads(output.getvalue().splitlines()[-1])
    assert end['progress'] == 'run_end' and end['status'] == 'failed'
    assert end['elapsed_sec'] >= 0


if __name__ == '__main__':
    test_elapsed_marker()
