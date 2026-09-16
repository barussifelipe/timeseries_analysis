from pathlib import Path
from tempfile import TemporaryDirectory

from inference import ranked_cells, save_summary_latex


def test_report_ranking():
    columns = ["split", "mse", "r2"]
    rows = [["a", 1.0, -0.2], ["b", 2.0, 0.1], ["c", 3.0, 0.0]]

    assert ranked_cells(columns, rows) == {
        (0, 1): "red", (1, 1): "blue",
        (1, 2): "red", (2, 2): "blue",
    }

    with TemporaryDirectory() as directory:
        path = Path(directory) / "summary.tex"
        save_summary_latex(columns, rows, path, window_size=30)
        latex = path.read_text(encoding="utf-8")
        assert r"\usepackage{xcolor}" in latex
        assert r"\begin{table}[H]" in latex
        assert r"\resizebox{\textwidth}{!}" in latex
        assert "window size 30" in latex
        assert r"\label{tab:lstm-final-summary-ws30}" in latex
        assert r"\textcolor{red}{\textbf{0.1}}" in latex
        assert r"\textcolor{blue}{\textbf{0}}" in latex


if __name__ == "__main__":
    test_report_ranking()
    print("report ranking check passed")
