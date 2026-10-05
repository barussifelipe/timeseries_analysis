"""Plot raw variance forecast error boxplots for the ten retained model variants."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

MODELS = (
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
)


def main():
    root = Path(__file__).resolve().parents[1] / 'imgs' / 'eval'
    data = []
    names = []

    for name, label, window in MODELS:
        rows = []
        forecast_file = root / 'mcs' / f'{window}_size' / 'forecasts.csv'
        with forecast_file.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if row['candidate'] == label:
                    rows.append(row)
        if len(rows) != 8800:
            raise ValueError(f'{label}: expected 8,800 rows, got {len(rows)}')
        actual = np.array([float(r['actual']) for r in rows])
        predicted = np.array([float(r['predicted']) for r in rows])
        raw_res = actual - predicted
        data.append(raw_res)
        names.append(name)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot(
        data, tick_labels=names, orientation='horizontal', whis=1.5,
        flierprops=dict(marker='.', markersize=2, alpha=0.2)
    )
    ax.axvline(0, color='0.4', linewidth=1, linestyle='--')
    ax.set_xlabel(r'Raw forecast error $e_i(t) = v_i(t) - \hat{v}_i(t)$')
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='x', alpha=0.2)
    ax.invert_yaxis()
    fig.tight_layout()
    output_path = root / 'local_eval' / 'residuals' / 'residual_boxplot_raw.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f'Successfully saved boxplot to {output_path}')


if __name__ == '__main__':
    main()
