import sys 
import argparse
import gc
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from training import * 
import wandb
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import FEBLSTM
from data.data_fetching import (
    HISTORY_DB,
    TimeSeriesDataset,
    load_return_split,
    prepare_return_tickers,
    fit_zscore_stats,
    apply_zscore,
)

METRIC_RANKING = {
    "mse": False, "rmse": False, "mae": False, "smape": False, "r2": True,
}


def ranked_cells(columns, rows):
    ranked = {}
    for column, reverse in METRIC_RANKING.items():
        if column not in columns:
            continue
        column_index = columns.index(column)
        row_indices = sorted(
            range(len(rows)),
            key=lambda index: rows[index][column_index],
            reverse=reverse,
        )
        for row_index, color in zip(row_indices[:2], ("red", "blue")):
            ranked[row_index, column_index] = color
    return ranked


def save_table_image(columns, rows, path, title):
    display_rows = [
        [f"{value:.6g}" if isinstance(value, float) else str(value) for value in row]
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(16, 1.2 + 0.42 * len(rows)))
    axis.axis("off")
    table = axis.table(cellText=display_rows, colLabels=columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row_index, column_index), color in ranked_cells(columns, rows).items():
        text = table[(row_index + 1, column_index)].get_text()
        text.set_color(color)
        text.set_weight("bold")
    axis.set_title(f"{title}\nRed = best; blue = second best", pad=12)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_histogram(values, path, title, xlabel):
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=50)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Ticker frequency")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_summary_latex(columns, rows, path):
    alignments = "l" + "r" * (len(columns) - 1)
    colors = ranked_cells(columns, rows)

    def format_cell(row_index, column_index, value):
        formatted = f"{value:.6g}" if isinstance(value, float) else str(value)
        color = colors.get((row_index, column_index))
        return rf"\textcolor{{{color}}}{{\textbf{{{formatted}}}}}" if color else formatted

    lines = [
        r"% Requires \usepackage{xcolor}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Final metrics from the best-validation LSTM checkpoint. Red denotes the best value and blue the second best.}",
        r"\label{tab:lstm-final-summary}",
        rf"\begin{{tabular}}{{{alignments}}}",
        r"\hline",
        " & ".join(columns) + r" \\",
        r"\hline",
    ]
    lines.extend(
        " & ".join(
            format_cell(row_index, column_index, value)
            for column_index, value in enumerate(row)
        ) + r" \\"
        for row_index, row in enumerate(rows)
    )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_splits(model, criterion, datasets, batch_size, device):
    final_metrics = {}
    ticker_metrics = {}
    for split, dataset in datasets.items():
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        metrics, per_ticker = test_batch(
            model, criterion, loader, device, type=split,
            per_ticker=split != "train",
        )
        final_metrics[split] = metrics
        ticker_metrics[split] = per_ticker
    return final_metrics, ticker_metrics


def build_final_report(checkpoint_path, best_epoch, datasets, final_metrics, ticker_metrics):
    metric_names = ("mse", "rmse", "mae", "smape", "r2")
    summary_columns = ["split", "observations", "tickers", *metric_names]
    summary_rows = []
    final_logs = {}
    final_values = {"best_epoch": best_epoch}
    for split, dataset in datasets.items():
        values = final_metrics[split]
        summary_rows.append([
            split,
            len(dataset),
            torch.unique(dataset.ticker_ids).numel(),
            *(values[f"{split}/{metric}"] for metric in metric_names),
        ])
        final_values.update({f"final/{key}": value for key, value in values.items()})

    report_dir = Path(checkpoint_path).with_suffix("")
    report_dir.mkdir(exist_ok=True)
    save_summary_latex(summary_columns, summary_rows, report_dir / "final_summary.tex")
    final_summary = wandb.Table(columns=summary_columns, data=summary_rows)
    final_logs["final_summary"] = final_summary
    ticker_columns = ["ticker", "observations", *metric_names]
    for split in ("val", "test"):
        rows = ticker_metrics[split]
        best = sorted(rows, key=lambda row: row["mse"])[:15]
        best_rows = [[row[column] for column in ticker_columns] for row in best]
        final_logs[f"{split}_best_15_tickers"] = wandb.Table(
            columns=ticker_columns,
            data=best_rows,
        )
        table_path = report_dir / f"{split}_best_15_tickers.png"
        save_table_image(
            ticker_columns, best_rows, table_path,
            f"{split.title()} best 15 tickers by MSE",
        )
        final_logs[f"{split}_best_15_tickers_image"] = wandb.Image(str(table_path))
        for metric in metric_names:
            values = [row[metric] for row in rows if math.isfinite(row[metric])]
            ordered = sorted(values)
            lower = ordered[math.floor(0.005 * (len(ordered) - 1))]
            upper = ordered[math.ceil(0.995 * len(ordered)) - 1]
            central = [value for value in values if lower <= value <= upper]
            for plotted, title, suffix in (
                (values, "full range", "full"),
                (central, "central 99%", "central_99"),
            ):
                histogram_path = report_dir / f"{split}_{metric}_{suffix}.png"
                save_histogram(
                    plotted, histogram_path,
                    f"{split.title()} {metric.upper()} — {title}", metric.upper(),
                )
                final_logs[f"{split}_{metric}_{suffix}"] = wandb.Image(str(histogram_path))
            final_values[f"{split}_{metric}_central_lower"] = lower
            final_values[f"{split}_{metric}_central_upper"] = upper
            final_values[f"{split}_{metric}_outliers"] = len(values) - len(central)

    return final_logs, final_values, summary_columns, summary_rows, report_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--window-size", type=int, choices=(10, 30, 100), default=30)
    parser.add_argument("--run-name")
    args = parser.parse_args()

    conn = sqlite3.connect(HISTORY_DB)

    epochs = args.epochs
    learning_rate = 1e-4
    batch_size = 128
    hidden_size = 64
    window_size = args.window_size
    patience = args.patience
    seed = 42
    run_name = args.run_name or f"returns_bs{batch_size}_hs{hidden_size}_ws{window_size}_lr{learning_rate}_epochs{epochs}_patience{patience}"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run = wandb.init(
        project="timeseries analysis",
        entity="personalfeb",  
        name=run_name,
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "window_size": window_size,
            "patience": patience,
            "learning_rate_reduction_factor": 0.1,
            "ticker_selection": "complete 20y coverage",
            "data_start": "2006-01-01",
            "data_end_exclusive": "2026-01-01",
            "target": "next-day overnight_returns",
            "seed": seed,
        }
    )
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("learning_rate", step_metric="epoch")
    print(f"Initializing model with hidden size: {hidden_size}, window size: {window_size}, learning rate: {learning_rate}, epochs: {epochs}, batch size: {batch_size}")
    type_return = 'overnight_returns'
    ticker_count = prepare_return_tickers(conn)
    print(f"Selected {ticker_count} tickers. Loading train split...")
    df_train = load_return_split(conn, '2006-01-01', '2016-01-01')

    volume_stats = fit_zscore_stats(df_train, column='log_volume')
    df_train = apply_zscore(df_train, volume_stats, column='log_volume')
    print(f"log_volume normalized. Train mean: {df_train['log_volume'].mean():.4f}, std: {df_train['log_volume'].std():.4f}")
    train_dataset = TimeSeriesDataset(df_train, window_size=window_size, type_return=type_return)
    del df_train
    gc.collect()

    print("Loading validation split...")
    df_val = apply_zscore(
        load_return_split(conn, '2016-01-01', '2019-01-01'),
        volume_stats,
        column='log_volume',
    )
    val_dataset = TimeSeriesDataset(df_val, window_size=window_size, type_return=type_return)
    del df_val
    gc.collect()

    print("Loading test split...")
    df_test = apply_zscore(
        load_return_split(conn, '2019-01-01', '2026-01-01'),
        volume_stats,
        column='log_volume',
    )
    test_dataset = TimeSeriesDataset(df_test, window_size=window_size, type_return=type_return)
    del df_test
    gc.collect()
    conn.close()

    print(f"Datasets created. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    run.config.update({
        "features": train_dataset.feature_columns,
        "selected_tickers": ticker_count,
        "train_observations": len(train_dataset),
        "val_observations": len(val_dataset),
        "test_observations": len(test_dataset),
    })

    model = FEBLSTM(input_size=len(train_dataset.feature_columns), hidden_size=hidden_size, output_size=1)    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model = torch.compile(model)  # Compile the model for optimized performance

    parameters(model)  # Print model parameters and memory usage

    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print(f"Starting training...")
    checkpoint_path = train(
        model, train_dataset, val_dataset, optimizer, criterion,
        epochs, batch_size, device, name_run=run_name, patience=patience,
    )
    model, best_epoch, _, _ = load_model(model, checkpoint_path)

    datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    final_metrics, ticker_metrics = evaluate_splits(
        model, criterion, datasets, batch_size, device,
    )
    final_logs, final_values, summary_columns, summary_rows, report_dir = build_final_report(
        checkpoint_path, best_epoch, datasets, final_metrics, ticker_metrics,
    )

    run.log(final_logs)
    run.summary.update(final_values)
    run.summary["final_summary"] = final_summary
    report_artifact = wandb.Artifact(f"{run.id}-evaluation", type="evaluation")
    report_artifact.add_dir(str(report_dir))
    run.log_artifact(report_artifact)
    run.finish()

    print(f"Final report saved to {report_dir}")
    print("Final summary:")
    print(" | ".join(summary_columns))
    for row in summary_rows:
        print(" | ".join(f"{value:.6g}" if isinstance(value, float) else str(value) for value in row))


    
