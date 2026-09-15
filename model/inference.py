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
from model import FEBLSTM
from data.data_fetching import (
    HISTORY_DB,
    TimeSeriesDataset,
    load_return_split,
    prepare_return_tickers,
    fit_zscore_stats,
    apply_zscore,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window-size", type=int, choices=(10, 30, 100), default=30)
    parser.add_argument("--ticker-limit", type=int, default=3000)
    parser.add_argument("--run-name")
    args = parser.parse_args()

    conn = sqlite3.connect(HISTORY_DB)

    epochs = args.epochs
    learning_rate = 1e-4
    batch_size = 128
    hidden_size = 64
    window_size = args.window_size
    seed = 42
    run_name = args.run_name or f"returns_bs{batch_size}_hs{hidden_size}_ws{window_size}_lr{learning_rate}_epochs{epochs}"

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
            "ticker_limit": args.ticker_limit,
            "ticker_selection": "most observations, then ticker name",
            "data_start": "2006-01-01",
            "data_end_exclusive": "2026-01-01",
            "target": "next-day overnight_returns",
            "seed": seed,
        }
    )
    print(f"Initializing model with hidden size: {hidden_size}, window size: {window_size}, learning rate: {learning_rate}, epochs: {epochs}, batch size: {batch_size}")
    type_return = 'overnight_returns'
    ticker_count = prepare_return_tickers(conn, args.ticker_limit)
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
        epochs, batch_size, device, name_run=run_name,
    )
    model, best_epoch, _, _ = load_model(model, checkpoint_path)

    final_metrics = {}
    ticker_metrics = {}
    for split, dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
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

    metric_names = ("mse", "rmse", "mae", "smape", "r2")
    summary_rows = []
    final_logs = {"best_epoch": best_epoch}
    for split, dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        values = final_metrics[split]
        summary_rows.append([
            split,
            len(dataset),
            torch.unique(dataset.ticker_ids).numel(),
            *(values[f"{split}/{metric}"] for metric in metric_names),
        ])
        final_logs.update({f"final/{key}": value for key, value in values.items()})

    final_logs["final/loss_table"] = wandb.Table(
        columns=["split", "observations", "tickers", *metric_names],
        data=summary_rows,
    )
    ticker_columns = ["ticker", "observations", *metric_names]
    for split in ("val", "test"):
        rows = ticker_metrics[split]
        best = sorted(rows, key=lambda row: row["mse"])[:15]
        final_logs[f"final/{split}_best_15_tickers"] = wandb.Table(
            columns=ticker_columns,
            data=[[row[column] for column in ticker_columns] for row in best],
        )
        for metric in metric_names:
            values = [row[metric] for row in rows if math.isfinite(row[metric])]
            final_logs[f"final/{split}_{metric}_ticker_distribution"] = wandb.Histogram(values)

    run.log(final_logs)
    run.finish()

    print("Final metrics:", final_metrics)


    

