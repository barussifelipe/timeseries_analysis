"""Verify thesis reporting against saved artifacts without rescoring forecasts."""
import csv
import math
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'ref/final_report/thesis_structure.tex').read_text(encoding='utf-8')
metrics = ['MAE', 'MASE', 'MSE', 'RMSE', 'QLIKE']
global_scales = [1e4, 1, 1e5, 1e3, 1]
local_scales = [1e4, 1, 1e7, 1e4, 1]
crypto_scales = [1e3, 1, 1e4, 1e2, 1]
counts = {20: 4479681, 80: 4295773}
rows = []
displayed = {}
result_tables = source.split(r'\section{Technical Evaluation}', 1)[1]
for file in ['imgs/eval/volatility_test_metrics.csv', 'imgs/eval/local_eval/metrics.csv']:
    scales = global_scales if file.endswith('volatility_test_metrics.csv') else local_scales
    for row in csv.DictReader((root / file).open()):
        name = row['model'].replace(' LOCAL', ' Local').replace(' GLOBAL', ' Global')
        line = next(s for s in result_tables.splitlines() if s.startswith(name + ' &'))
        cells = [s.strip().removesuffix('\\\\') for s in line.split(' & ')]
        assert len(cells) == 6, name
        displayed[name] = cells[1:]
        for cell, key, scale in zip(cells[1:], metrics, scales):
            cell = re.sub(r'\\textcolor\{(?:red|blue|orange!65!black)\}\{([^}]+)\}', r'\1', cell)
            assert math.isclose(float(cell), float(row[key])*scale, rel_tol=5e-6), (name, key)
        expected_n = counts[80 if row['model'].endswith('-80') else 20]
        assert int(row.get('N', expected_n)) == (8800 if row.get('population') else expected_n)
        assert math.isclose(float(row['RMSE'])**2, float(row['MSE']), rel_tol=1e-12)
        if row.get('population') == 'full-period':
            rows.append(row)
full = rows
assert len(displayed) == 45
assert source.count(r'\begin{longtable}{@{}p{.235\linewidth}>{\raggedleft\arraybackslash}p{\dimexpr .165\linewidth-2\tabcolsep\relax}>{\raggedleft\arraybackslash}p{\dimexpr .115\linewidth-2\tabcolsep\relax}>{\raggedleft\arraybackslash}p{\dimexpr .165\linewidth-2\tabcolsep\relax}>{\raggedleft\arraybackslash}p{\dimexpr .175\linewidth-2\tabcolsep\relax}>{\raggedleft\arraybackslash}p{\dimexpr .145\linewidth-2\tabcolsep\relax}@{}}') == 4
assert source.count(r'MSE ($\times10^{-5}$)') == 2
assert source.count(r'MSE ($\times10^{-7}$)') == 2
assert source.count(r'RMSE ($\times10^{-3}$)') == 2
assert source.count(r'RMSE ($\times10^{-4}$)') == 2
assert source.count(r'MAE ($\times10^{-3}$)') == 4
assert source.count(r'& MSE ($\times10^{-4}$)') == 2
assert source.count(r'& MSE ($\times10^{-2}$)') == 2
assert source.count(r'RMSE ($\times10^{-2}$)') == 2
assert source.count(r'RMSE ($\times10^{-1}$)') == 2
assert not re.search(r'\\hline\s+RFSV-20 Local', source)
assert 'Model & $N$' not in source and 'Floor (\\%)' not in source
for names in [
    [r['model'] for r in csv.DictReader((root / 'imgs/eval/volatility_test_metrics.csv').open())],
    [r['model'].replace(' LOCAL', ' Local').replace(' GLOBAL', ' Global') for r in csv.DictReader((root / 'imgs/eval/local_eval/metrics.csv').open())],
]:
    for index in range(5):
        ranked = sorted(names, key=lambda name: float(re.sub(r'\\textcolor\{(?:red|blue|orange!65!black)\}\{([^}]+)\}', r'\1', displayed[name][index])))
        assert displayed[ranked[0]][index].startswith(r'\textcolor{red}{')
        assert displayed[ranked[1]][index].startswith(r'\textcolor{blue}{')
        assert displayed[ranked[-1]][index].startswith(r'\textcolor{orange!65!black}{')
        assert sum(r'\textcolor{red}{' in displayed[name][index] for name in names) == 1
        assert sum(r'\textcolor{blue}{' in displayed[name][index] for name in names) == 1
        assert sum(r'\textcolor{orange!65!black}{' in displayed[name][index] for name in names) == 1
assert not any(name.startswith('RFSV-full') for name in displayed)
assert len(full) == 30
crypto = list(csv.DictReader((root / 'imgs/eval/crypto_eval/metrics.csv').open()))
crypto_table = source.split(r'\label{tab:matched-results}', 1)[1].split(r'\label{tab:crypto-results}', 1)[0]
crypto_displayed = {}
for row in crypto:
    name = row['model'].removesuffix(' GLOBAL')
    line = next(s for s in crypto_table.splitlines() if s.startswith(name + ' &'))
    cells = [s.strip().removesuffix('\\\\') for s in line.split(' & ')]
    assert len(cells) == 6 and int(row['N']) == 8800
    crypto_displayed[name] = cells[1:]
    for cell, key, scale in zip(cells[1:], metrics, crypto_scales):
        number = re.sub(r'\\textcolor\{(?:red|blue|orange!65!black)\}\{([^}]+)\}', r'\1', cell)
        assert math.isclose(float(number), float(row[key])*scale, rel_tol=5e-6), (name, key)
    assert math.isclose(float(row['RMSE'])**2, float(row['MSE']), rel_tol=1e-12)
assert len(crypto_displayed) == 15
for index in range(5):
    ranked = sorted(crypto_displayed, key=lambda name: float(re.sub(
        r'\\textcolor\{(?:red|blue|orange!65!black)\}\{([^}]+)\}', r'\1', crypto_displayed[name][index])))
    for name, color in ((ranked[0], 'red'), (ranked[1], 'blue'), (ranked[-1], 'orange!65!black')):
        assert crypto_displayed[name][index].startswith(r'\textcolor{' + color + '}{')
        assert sum(r'\textcolor{' + color + '}{' in cells[index] for cells in crypto_displayed.values()) == 1

crypto_rv = list(csv.DictReader((root / 'imgs/eval/crypto_rv_eval/metrics.csv').open()))
crypto_rv_table = source.split(r'\label{tab:crypto-results}', 1)[1].split(r'\label{tab:crypto-rv-results}', 1)[0]
crypto_rv_displayed = {}
crypto_rv_scales = [1e3, 1, 1e2, 1e1, 1]
for row in crypto_rv:
    name = row['model'].removesuffix(' GLOBAL')
    line = next(s for s in crypto_rv_table.splitlines() if s.startswith(name + ' &'))
    cells = [s.strip().removesuffix(r'\\') for s in line.split(' & ')]
    assert len(cells) == 6 and int(row['N']) == 8800
    crypto_rv_displayed[name] = cells[1:]
    for cell, key, scale in zip(cells[1:], metrics, crypto_rv_scales):
        number = re.sub(r'\\textcolor\{[^}]+\}\{(.*)\}', r'\1', cell)
        if r'\times10^{' in number:
            m = re.match(r'\$?([0-9.]+)\\times10\^\{([0-9]+)\}\$?', number)
            assert m, (name, key, number)
            val = float(m.group(1)) * (10 ** int(m.group(2)))
            assert math.isclose(val, float(row[key])*scale, rel_tol=1e-2), (name, key)
        else:
            assert math.isclose(float(number), float(row[key])*scale, rel_tol=5e-6), (name, key)
    assert math.isclose(float(row['RMSE'])**2, float(row['MSE']), rel_tol=1e-12)
assert len(crypto_rv_displayed) == 15
for index in range(5):
    def parse_cell(c):
        clean = re.sub(r'\\textcolor\{[^}]+\}\{(.*)\}', r'\1', c)
        if r'\times10^{' in clean:
            m = re.match(r'\$?([0-9.]+)\\times10\^\{([0-9]+)\}\$?', clean)
            return float(m.group(1)) * (10 ** int(m.group(2)))
        return float(clean)
    ranked = sorted(crypto_rv_displayed, key=lambda name: parse_cell(crypto_rv_displayed[name][index]))
    for name, color in ((ranked[0], 'red'), (ranked[1], 'blue'), (ranked[-1], 'orange!65!black')):
        assert crypto_rv_displayed[name][index].startswith(r'\textcolor{' + color + '}{')
        assert sum(r'\textcolor{' + color + '}{' in cells[index] for cells in crypto_rv_displayed.values()) == 1

section = source.split(r'\section{Technical Evaluation}')[1].split(r'\section{Conclusions}')[0]
assert re.findall(r'\\subsection\{([^}]+)\}', section) == ['Overall Result', 'Crypto Evaluation', 'Model Statistical Significance', 'Residual Structure', 'Parameter Complexity']
assert re.findall(r'\\subsubsection\{([^}]+)\}', section) == ['Base LSTM', 'HARNet', 'MLP', 'RFSV']
paths = re.findall(r'\\includegraphics\[[^]]+\]\{([^}]+)\}', section)
variants = [('base_lstm_20', 'global'), ('base_lstm_80', 'global'), ('base_lstm_80', 'local'),
            ('harnet_20', 'global'), ('harnet_80', 'global'), ('harnet_80', 'local'),
            ('mlp_20', 'global'), ('mlp_80', 'global'), ('rfsv_20', 'local'), ('rfsv_80', 'local')]
test_paths = [f'../../imgs/eval/local_eval/test_data/{plot}.png'
              for plot in ('timeline_raw', 'timeline_log', 'histogram_raw', 'histogram_log')]
expected_paths = test_paths + [f'../../imgs/eval/local_eval/residuals/{model}/{scope}/{plot}.png'
                  for model, scope in variants
                  for plot in ('timeline_mean_raw', 'timeline_mean_log', 'histogram_raw', 'histogram_log')]
assert paths == expected_paths
assert len(paths) == len(set(paths)) == 44
assert all((root / 'ref/final_report' / s).is_file() for s in paths)
assert section.count(r'\begin{figure}[htbp]') == 22
assert section.count('the orange curve is a full-sample normal fit.') == 10
assert 'Each histogram has an orange normal curve fitted to all 8,800 pooled residuals using their mean and population variance.' in section
for model, scope in variants:
    family, window = model.rsplit('_', 1)
    name = f'{dict(base_lstm="Base LSTM", harnet="HARNet", mlp="MLP", rfsv="RFSV")[family]}-{window} {scope.title()}'
    assert section.count(r'\textbf{' + name + '.}') == 1
    slug = model.replace('_', '-') + '-' + scope
    assert section.count(r'\caption{' + name + ': raw and log daily cross-ticker mean residual timelines') == 1
    assert section.count(r'\caption{' + name + ': raw and log individual-residual histograms') == 1
    assert section.count(r'\label{fig:residual-' + slug + '-timeline}') == 1
    assert section.count(r'\label{fig:residual-' + slug + '-histograms}') == 1
figures = re.findall(r'\\begin\{figure\}\[htbp\](.*?)\\end\{figure\}', section, re.S)
assert len(figures) == 22
for index, figure in enumerate(figures):
    image_names = re.findall(r'/([^/]+\.png)\}', figure)
    expected_names = (['timeline_raw.png', 'timeline_log.png'] if index == 0 else
                      ['timeline_mean_raw.png', 'timeline_mean_log.png'] if index % 2 == 0
                      else ['histogram_raw.png', 'histogram_log.png'])
    assert image_names == expected_names
complexity_table = section.split(r'\label{tab:parameter-combinations}')[0].split(r'\begin{tabular}')[-1]
assert complexity_table.startswith('{lrrr}')
assert 'Model & $p$ & Global & Local (extrapolated)' in complexity_table
for p in [2,4,6,3,5,22021,37381,67201,13,19]:
    assert f'& {p:,} & $2^{{{64*p:,}}}$ & $2^{{{64*2714*p:,}}}$' in complexity_table
assert '$B^' not in complexity_table
assert '$(2^{64})' not in complexity_table
assert 'Global/Local' not in complexity_table
assert all(row.count('&') == 3 for row in complexity_table.splitlines() if '&' in row)
assert re.search(r'\\label\{tab:parameter-combinations\}\s*\\end\{table\}\s*\\FloatBarrier\s*Counts include fitted coefficients', section)
assert '\\subsection{Residual Structure}' in section
assert '\\numberwithin{table}{section}' in source
assert '\\numberwithin{figure}{section}' in source
before_file = root / 'judge/technical_evaluation_before.tex'
if before_file.exists():
    before = before_file.read_text(encoding='utf-8')
    prefix = source.split(r'\subsection{Evaluation Setup}')[0]
    prefix = prefix.replace('\\usepackage{graphicx}\n','').replace('\\usepackage{longtable}\n','')
    assert prefix == before.split(r'\subsection{Evaluation Setup}')[0]
    assert source.split(r'\section{Conclusions}')[1] == before.split(r'\section{Conclusions}')[1]
    assert source.split(r'\section{Experimental Results}')[1].split(r'\subsection{Computational Evaluation}')[0] == before.split(r'\section{Experimental Results}')[1].split(r'\subsection{Computational Evaluation}')[0]
print('PASS: saved metric rows, rankings, precision, RMSE identities; complexity exponents; two test-target figures and 20 labeled residual figures; subsection order.')
