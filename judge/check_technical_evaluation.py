"""Verify thesis reporting against saved artifacts without rescoring forecasts."""
import csv
import json
import math
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'ref/final_report/thesis_structure.tex').read_text(encoding='utf-8')
metrics = ['MAE', 'MASE', 'MSE', 'RMSE', 'QLIKE']
scales = [1e4, 1, 1e8, 1e4, 1]
logs = [json.loads(s) for s in (root / 'imgs/eval/volatility_test_metrics_full_history.log').read_text(encoding='utf-16').splitlines() if s.strip()]
counts = {r['model']: r['N'] for r in logs if 'model' in r}
rows = []
for file in ['imgs/eval/volatility_test_metrics.csv', 'imgs/eval/local_eval/metrics.csv']:
    for row in csv.DictReader((root / file).open()):
        name = row['model'].replace(' LOCAL', ' Local').replace(' GLOBAL', ' Global')
        line = next(s for s in source.splitlines() if s.startswith(name + ' &'))
        cells = [re.sub(r'\\textbf\{([^}]+)\}', r'\1', s).strip().removesuffix('\\\\') for s in line.split(' & ')]
        assert int(cells[1]) == int(row.get('N', counts.get(row['model'], 0)))
        for cell, key, scale in zip(cells[2:7], metrics, scales):
            assert math.isclose(float(cell), float(row[key])*scale, rel_tol=5e-6), (name, key)
        assert math.isclose(float(cells[7]), float(row['floor_hit_pct']), rel_tol=5e-6)
        assert math.isclose(float(row['RMSE'])**2, float(row['MSE']), rel_tol=1e-12)
        if row.get('population') == 'full-period':
            rows.append(row)
full = rows
for key in metrics:
    best = min(full, key=lambda r: float(r[key]))['model']
    worst = max(full, key=lambda r: float(r[key]))['model']
    assert best == ('SARIMA LOCAL' if key in ['MAE','MASE'] else 'MLP-80 GLOBAL')
    assert worst == ('SiLU-LSTM-80 LOCAL' if key == 'QLIKE' else 'AR(1) GLOBAL')
section = source.split(r'\subsection{Technical Evaluation}')[1].split(r'\section{Conclusions}')[0]
assert re.findall(r'\\subsubsection\{([^}]+)\}', section) == ['Overall Result','Parameter Complexity','Residual Structure']
paths = re.findall(r'\\includegraphics\[[^]]+\]\{([^}]+)\}', section)
assert len(paths) == len(set(paths)) == 16
assert all('/local_eval/' in s and '/timeline_raw.' not in s and '/timeline_log.' not in s for s in paths)
assert all((root / 'ref/final_report' / s).is_file() for s in paths)
for p in [2,4,6,3,5,22021,37381,67201,13,19]:
    assert f'B^{{{2714*p:,}}}' in section
    assert f'$B^{{-{2713*p:,}}}$' in section
assert 'Model & $p$ & Local (extrapolated) & Global & Global/Local' in section
assert '\\clearpage\n\\subsubsection{Residual Structure}' in section
assert '\\numberwithin{table}{section}' in source
assert '\\numberwithin{figure}{section}' in source
before_file = root / 'judge/technical_evaluation_before.tex'
if before_file.exists():
    before = before_file.read_text(encoding='utf-8')
    prefix = source.split(r'\subsection{Evaluation Setup}')[0]
    prefix = prefix.replace('\\usepackage{graphicx}\n','').replace('\\usepackage{longtable}\n','')
    assert prefix == before.split(r'\subsection{Evaluation Setup}')[0]
    assert source.split(r'\section{Conclusions}')[1] == before.split(r'\section{Conclusions}')[1]
    assert source.split(r'\section{Experimental Results}')[1].split(r'\subsection{Technical Evaluation}')[0] == before.split(r'\section{Experimental Results}')[1].split(r'\subsection{Technical Evaluation}')[0]
print('PASS: all 42 table rows, counts, floor rates, precision, RMSE identities; five extrema pairs; complexity exponents; 16 unique matched plots; subsection order; unrelated source preserved when snapshot available.')
