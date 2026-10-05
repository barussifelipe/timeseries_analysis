"""Verify residual underprediction and overprediction table in thesis_structure.tex."""

import csv
from pathlib import Path
import re
import numpy as np

MODELS = [
    ('Base LSTM-20 Global', 'Base LSTM-20 GLOBAL', 20),
    ('Base LSTM-80 Global', 'Base LSTM-80 GLOBAL', 80),
    ('Base LSTM-80 Local', 'Base LSTM-80 LOCAL', 80),
    ('HARNet-20 Global', 'HARNet-20 GLOBAL', 20),
    ('HARNet-80 Global', 'HARNet-80 GLOBAL', 80),
    ('HARNet-80 Local', 'HARNet-80 LOCAL', 80),
    ('MLP-20 Global', 'MLP-20 GLOBAL', 20),
    ('MLP-80 Global', 'MLP-80 GLOBAL', 80),
    ('RFSV-20 Local', 'RFSV-20 LOCAL', 20),
    ('RFSV-80 Local', 'RFSV-80 LOCAL', 80),
]

def main():
    root = Path(__file__).resolve().parents[1]
    tex_path = root / 'ref/final_report/thesis_structure.tex'
    source = tex_path.read_text(encoding='utf-8')

    res_section = source.split(r'\subsection{Residual Structure}', 1)[1].split(
        r'\subsection{Parameter Complexity}', 1
    )[0]

    # Verify brief descriptions are clean (no percentages inside them)
    for display_name, label, window in MODELS:
        desc_pattern = re.escape(r'\textbf{' + display_name + '.}') + r'\s+The saved \d+-observation (?:global|local) forecast is summarized in the following timeline and histogram figures\.'
        assert re.search(desc_pattern, res_section), f'Missing or modified brief description for {display_name}'

    # Verify table introduction describes the total number
    intro_pattern = r'Table~\\ref\{tab:residual-prediction-bias\}\s+reports the direction of one-step variance forecast errors across all 8,800 test predictions\s+\(1,760 daily observations across the five local stocks'
    assert re.search(intro_pattern, res_section), 'Missing or incorrect description before the table'

    # Extract the table
    table_match = re.search(r'\\begin\{table\}\[htbp\].*?\\label\{tab:residual-prediction-bias\}.*?\\end\{table\}', res_section, re.S)
    assert table_match, 'Could not find tab:residual-prediction-bias table'
    table_str = table_match.group(0)

    # Verify total (N) column is NOT in the table
    assert 'Total' not in table_str and '& $N$' not in table_str and '& N ' not in table_str, 'Total (N) column must be removed from the table'

    BEST_METRICS = {
        'Base LSTM-20 Global': 'QLIKE (2nd)',
        'Base LSTM-80 Global': 'QLIKE (4th)',
        'Base LSTM-80 Local': 'MASE (5th)',
        'HARNet-20 Global': 'QLIKE (6th)',
        'HARNet-80 Global': 'MSE (3rd)',
        'HARNet-80 Local': 'MAE/MASE (2nd)',
        'MLP-20 Global': 'QLIKE (8th)',
        'MLP-80 Global': 'MSE/QLIKE (1st)',
        'RFSV-20 Local': 'MSE (7th)',
        'RFSV-80 Local': 'MSE (2nd)',
    }

    # Verify Best Metric header in table
    assert 'Best Metric' in table_str, 'Best Metric column header missing from table'

    for display_name, label, window in MODELS:
        file = root / 'imgs/eval/mcs' / f'{window}_size' / 'forecasts.csv'
        actuals = []
        preds = []
        with file.open(newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['candidate'] == label:
                    actuals.append(float(r['actual']))
                    preds.append(float(r['predicted']))
        act = np.array(actuals)
        prd = np.array(preds)
        assert len(act) == 8800, f'{label}: expected 8800 actuals'
        assert len(prd) == 8800, f'{label}: expected 8800 preds'
        under = np.sum(act > prd)
        over = np.sum(act < prd)
        eq = np.sum(act == prd)
        assert eq == 0, f'{label}: expected 0 ties'
        pct_under = f'{under / len(act) * 100:.2f}'
        pct_over = f'{over / len(act) * 100:.2f}'

        # Check table row
        # Pattern: <display_name> & <best_metric> & <under:,> & <pct_under>\% & <over:,> & <pct_over>\% \\
        best_metric = BEST_METRICS[display_name]
        row_pattern = (
            re.escape(display_name)
            + r'\s*&\s*'
            + re.escape(best_metric)
            + r'\s*&\s*'
            + f'{under:,}'
            + r'\s*&\s*'
            + re.escape(pct_under)
            + r'\\%\s*&\s*'
            + f'{over:,}'
            + r'\s*&\s*'
            + re.escape(pct_over)
            + r'\\%'
        )
        match = re.search(row_pattern, table_str)
        assert match, f'Could not find matching table row for {display_name} with {best_metric} ({under:,}, {pct_under}%, {over:,}, {pct_over}%)'

    # Verify residual boxplot image and LaTeX figure
    boxplot_img = root / 'imgs/eval/local_eval/residuals/residual_boxplot_raw.png'
    assert boxplot_img.exists() and boxplot_img.stat().st_size > 10000, 'residual_boxplot_raw.png missing or too small'

    boxplot_fig_pattern = r'\\begin\{figure\}\[htbp\].*?\\includegraphics\[.*?\]\{.*?residual_boxplot_raw\.png\}.*?\\label\{fig:residual-boxplots-raw\}.*?\\end\{figure\}'
    assert re.search(boxplot_fig_pattern, res_section, re.S), 'fig:residual-boxplots-raw figure missing or malformed in thesis_structure.tex'

    # Verify moments.md
    moments_path = root / 'imgs/eval/local_eval/residuals/moments.md'
    assert moments_path.exists(), 'moments.md missing'
    moments_text = moments_path.read_text(encoding='utf-8')
    assert '| Model Variant | Plot / Transform | Mean | Std Dev | Skewness | Excess Kurtosis |' in moments_text, 'moments.md table header missing'
    assert 'residual_boxplot_raw.png' in moments_text, 'moments.md boxplot reference missing'

    print('PASS: Table and boxplot at end of Residual Structure verified: Best Metric column present, total described in preceding text, total (N) column omitted, boxplot figure and moments.md verified, and all 10 models match ground-truth under/overpredictions.')

if __name__ == '__main__':
    main()
