import sys 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from training import * 
import wandb
import sqlite3
from model import FEBLSTM
from data.data_fetching import TimeSeriesDataset, load_data


if __name__ == "__main__":
    conn = sqlite3.connect("data/src/stock_data.db")

    epochs = 50
    learning_rate = 1e-3
    batch_size = 32
    hidden_size = 64
    window_size = 30
    run_name = f"first_run_bs{batch_size}_hs{hidden_size}_ws{window_size}_lr{learning_rate}_epochs{epochs}"

    wandb.init(
        project="timeseries analysis",
        entity="personalfeb",  
        name=run_name,   # Optional: specific run name
        config={                            # Store model hyperparameters
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "window_size": window_size
        }
    )
    print(f"Initializing model with hidden size: {hidden_size}, window size: {window_size}, learning rate: {learning_rate}, epochs: {epochs}, batch size: {batch_size}")
    model = FEBLSTM(input_size=1, hidden_size=hidden_size, output_size=1)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading data from database...")
    df_train, df_val, df_test = load_data(conn, "intraday_returns")
    print(f"Data loaded. Train shape: {df_train.shape}, Val shape: {df_val.shape}, Test shape: {df_test.shape}")

    print(f"Creating datasets...")
    train_dataset = TimeSeriesDataset(df_train, window_size=window_size, type_return='intraday_returns')
    val_dataset = TimeSeriesDataset(df_val, window_size=window_size, type_return='intraday_returns')
    test_dataset = TimeSeriesDataset(df_test, window_size=window_size, type_return='intraday_returns')

    print(f"Datasets created. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    print(f"Starting training...")
    train_metrics, val_metrics = train(model, train_dataset, val_dataset, optimizer, criterion, epochs, batch_size, device, name_run=run_name)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_metrics = test_batch(model, criterion, test_loader, device, type="test")

    parameters_memory(model)

    wandb.log({"test_metrics": test_metrics})  # Log test metrics to wandb
    wandb.finish()  # Finish the wandb run

    print("Test Metrics:", test_metrics, "Train Metrics:", train_metrics, "Validation Metrics:", val_metrics)


    

