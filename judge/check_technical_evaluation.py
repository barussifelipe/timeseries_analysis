"""Verify thesis reporting against saved artifacts without rescoring forecasts."""
import csv
import math
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'ref/final_report/thesis_structure.tex').read_text(encoding='utf-8')
metrics = ['MAE', 'MASE', 'MSE', 'RMSE', 'QLIKE']
scales = [1e4, 1, 1e9, 1e4, 1]
counts = {20: 4479681, 80: 4295773}
rows = []
displayed = {}
for file in ['imgs/eval/volatility_test_metrics.csv', 'imgs/eval/local_eval/metrics.csv']:
    for row in csv.DictReader((root / file).open()):
        name = row['model'].replace(' LOCAL', ' Local').replace(' GLOBAL', ' Global')
        line = next(s for s in source.splitlines() if s.startswith(name + ' &'))
        cells = [s.strip().removesuffix('\\\\') for s in line.split(' & ')]
        assert len(cells) == 6, name
        displayed[name] = cells[1:]
        for cell, key, scale in zip(cells[1:], metrics, scales):
            cell = re.sub(r'\\textcolor\{(?:red|blue)\}\{([^}]+)\}', r'\1', cell)
            assert math.isclose(float(cell), float(row[key])*scale, rel_tol=5e-6), (name, key)
        expected_n = counts[80 if row['model'].endswith('-80') else 20]
        assert int(row.get('N', expected_n)) == (8800 if row.get('population') else expected_n)
        assert math.isclose(float(row['RMSE'])**2, float(row['MSE']), rel_tol=1e-12)
        if row.get('population') == 'full-period':
            rows.append(row)
full = rows
assert len(displayed) == 45
assert source.count(r'\begin{longtable}{@{}p{.235\linewidth}>{\raggedleft\arraybackslash}p{\dimexpr .165\linewidth-2\tabcolsep\relax}>{\raggedleft\arraybackslash}p{\dimexpr .115\linewidth-2\tabcolsep\relax}>{\raggedleft\arraybackslash}p{\dimexpr .165\linewidth-2\tabcolsep\relax}>{\raggedleft\arraybackslash}p{\dimexpr .175\linewidth-2\tabcolsep\relax}>{\raggedleft\arraybackslash}p{\dimexpr .145\linewidth-2\tabcolsep\relax}@{}}') == 2
assert source.count(r'MSE ($\times10^{-9}$)') == 4
assert source.count(r'RMSE ($\times10^{-4}$)') == 4
assert not re.search(r'\\hline\s+RFSV-20 Local', source)
assert 'Model & $N$' not in source and 'Floor (\\%)' not in source
for names in [
    [r['model'] for r in csv.DictReader((root / 'imgs/eval/volatility_test_metrics.csv').open())],
    [r['model'].replace(' LOCAL', ' Local').replace(' GLOBAL', ' Global') for r in csv.DictReader((root / 'imgs/eval/local_eval/metrics.csv').open())],
]:
    for index in range(5):
        ranked = sorted(names, key=lambda name: float(re.sub(r'\\textcolor\{(?:red|blue)\}\{([^}]+)\}', r'\1', displayed[name][index])))
        assert displayed[ranked[0]][index].startswith(r'\textcolor{red}{')
        assert displayed[ranked[1]][index].startswith(r'\textcolor{blue}{')
        assert sum(r'\textcolor{red}{' in displayed[name][index] for name in names) == 1
        assert sum(r'\textcolor{blue}{' in displayed[name][index] for name in names) == 1
assert not any(name.startswith('RFSV-full') for name in displayed)
assert len(full) == 30
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
print('PASS: all 45 saved metric rows, color rankings, precision, RMSE identities; complexity exponents; 16 unique matched plots; subsection order; unrelated source preserved when snapshot available.')
