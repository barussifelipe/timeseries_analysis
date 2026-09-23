"""Small report check; no database, W&B, or training required."""

import itertools
import tempfile
from pathlib import Path
from unittest.mock import patch

from models import lstm_val_sweep


def test_lstm_val_sweep():
    cases = list(itertools.product(('base_lstm_vol', 'silu_lstm'), (20, 40, 80),
                                   (16, 32, 64), (1e-3, 1e-4)))
    assert len(cases) == len(set(cases)) == 36
    with tempfile.TemporaryDirectory() as directory, patch.object(
            lstm_val_sweep, 'RESULTS', Path(directory) / 'results.md'):
        rows = [dict(name='limited_preflight', status='complete', log='limited.log',
                     metrics={'qlike': 99.}),
                dict(name='full', status='complete', log='full.log', epochs=2,
                     best_epoch=1, metrics={key: 1. for key in lstm_val_sweep.METRICS})]
        lstm_val_sweep.write_results(rows)
        report = lstm_val_sweep.RESULTS.read_text(encoding='utf-8')
        assert 'limited_preflight' not in report and '99' not in report
        assert '| full | complete | 2 | 1 |' in report


if __name__ == '__main__':
    test_lstm_val_sweep()
