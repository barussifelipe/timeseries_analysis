"""Plot the displayed Table 1 moment values across model variants."""

import re
from math import isclose
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


TABLE = Path(__file__).with_name('moments.md')
METRICS = ('Mean', 'Std Dev', 'Skewness', 'Excess Kurtosis')
SUPERSCRIPTS = str.maketrans('⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹', '-+0123456789')


def table_number(value):
    match = re.fullmatch(r'([+-]?\d+(?:\.\d+)?)\s*×\s*10([⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹]+)', value)
    return float(match[1]) * 10 ** int(match[2].translate(SUPERSCRIPTS)) if match else float(value)


def main():
    groups = {'Raw Residuals': [], 'Log Residuals': []}
    for line in TABLE.read_text(encoding='utf-8').splitlines():
        if not line.startswith('|'):
            continue
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) == 6 and cells[1] in groups:
            groups[cells[1]].append([table_number(value) for value in cells[2:]])

    for plot_type, rows in groups.items():
        if len(rows) != 10:
            raise ValueError(f'{plot_type}: expected ten Table 1 model variants, got {len(rows)}')
        transform = 'raw' if plot_type == 'Raw Residuals' else 'log'
        for index, metric in enumerate(METRICS):
            values = np.array([row[index] for row in rows])
            fig, ax = plt.subplots(figsize=(6, 4))
            _, edges, _ = ax.hist(values, bins=5, color='#236c93', edgecolor='white',
                                  label='Table values')
            mean, std = values.mean(), values.std(ddof=0)
            if not np.isfinite(std) or std <= 0:
                raise ValueError(f'{plot_type} {metric}: normal fit needs positive spread')
            x = np.linspace(edges[0], edges[-1], 400)
            normal_counts = (len(values) * (edges[1] - edges[0])
                             * np.exp(-0.5 * ((x - mean) / std) ** 2)
                             / (std * np.sqrt(2 * np.pi)))
            ax.plot(x, normal_counts, color='#bc4b2f', linewidth=2,
                    label='Normal fit to 10 values')
            ax.set(title=f'{plot_type}: {metric} across models',
                   xlabel=metric, ylabel='Model variants (N = 10)')
            ax.grid(axis='y', alpha=.2)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.legend()
            fig.tight_layout()
            fig.savefig(TABLE.with_name(f'moments_{transform}_{metric.lower().replace(" ", "_")}.png'), dpi=150)
            plt.close(fig)


if __name__ == '__main__':
    assert isclose(table_number('-4.293 × 10⁻⁵'), -4.293e-5)
    assert table_number('+87.18') == 87.18
    main()
