"""Check the thesis MCS table and common set against saved cross-window results."""

import json
import re
from pathlib import Path


root = Path(__file__).resolve().parents[1]
source = (root / 'ref/final_report/thesis_structure.tex').read_text(encoding='utf-8')
section = source.split(r'\subsubsection{Model statistical significance}', 1)[1].split(
    r'\subsubsection{Parameter Complexity}', 1)[0]
reports = {block: json.loads((root / f'imgs/eval/mcs/all_windows/mcs_{block}.json').read_text(
    encoding='utf-8')) for block in (20, 80)}
table = section.split(r'\label{tab:mcs-cross-window}', 1)[0]
for ticker in sorted(reports[20]['stocks']):
    sets = {block: set(reports[block]['stocks'][ticker]['members']) for block in (20, 80)}
    added = sets[80] - sets[20]
    removed = sets[20] - sets[80]
    assert not removed
    difference = 'None' if not added else ', '.join(
        name.rsplit(' ', 1)[0] + ' ' + name.rsplit(' ', 1)[1].title()
        for name in sorted(added)) + ' in MCS-80 only'
    row = rf'{ticker} & {len(sets[20])} & {len(sets[80])} & {re.escape(difference)}\\'
    assert len(re.findall(row, table)) == 1, ticker

common = {block: set.intersection(*(set(stock['members']) for stock in
                                    reports[block]['stocks'].values())) for block in (20, 80)}
assert common[20] == common[80]
listed = section.split('both} block lengths: ', 1)[1].split('. Retention', 1)[0]
listed = {name.removeprefix('and ') for name in listed.split('; ')}
expected = {name.rsplit(' ', 1)[0] + ' ' + name.rsplit(' ', 1)[1].title()
            for name in common[20]}
assert listed == expected
assert r'\FloatBarrier' in section.split(r'\end{table}', 1)[1]
print('MCS thesis counts, differences, shared set, and float placement match saved results')
