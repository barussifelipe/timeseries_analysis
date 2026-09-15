from model import *
import wandb
import torch 
from torch.utils.data import DataLoader, TensorDataset
import os 
import math





def losses(outputs, labels, epsilon=1e-8):
    """Return additive totals used to calculate exact epoch metrics."""
    error = outputs - labels
    absolute_error = torch.abs(error)
    # sMAPE replaces MAPE because signed financial returns frequently equal or
    # approach zero. It is symmetric and bounded between 0% and 200%.
    smape = 200 * absolute_error / (torch.abs(outputs) + torch.abs(labels) + epsilon)
    return {
        "observations": labels.numel(),
        "absolute_error": absolute_error.sum(),
        "squared_error": torch.square(error).sum(),
        "smape": smape.sum(),
        "target_sum": labels.sum(),
        "target_squared_sum": torch.square(labels).sum(),
    }


def _empty_totals():
    return {
        "observations": 0,
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "smape": 0.0,
        "target_sum": 0.0,
        "target_squared_sum": 0.0,
    }


def _add_totals(total, batch):
    for key, value in batch.items():
        total[key] += value if isinstance(value, int) else value.item()


def _metrics(total, prefix):
    observations = total["observations"]
    if observations == 0:
        return {f"{prefix}/{name}": float("nan") for name in ("mse", "rmse", "mae", "smape", "r2")}

    mse = total["squared_error"] / observations
    target_variance = (
        total["target_squared_sum"]
        - total["target_sum"] ** 2 / observations
    )
    values = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": total["absolute_error"] / observations,
        "smape": total["smape"] / observations,
        "r2": (
            1 - total["squared_error"] / target_variance
            if target_variance > 0 else float("nan")
        ),
    }
    return {f"{prefix}/{name}": value for name, value in values.items()}

def train_batch(model, optimizer, criterion, data_loader, device):
    """
    Trains the model on a single batch of data.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (nn.Module): The loss function to compute the loss.
        data_loader (DataLoader): The DataLoader providing batches of training data.
        device (torch.device): The device to run the training on (CPU or GPU).

    Returns:
        float: The computed loss for the current batch.
    """
    num_batches = 0
    totals = _empty_totals()

    model.train()  # Set the model to training mode
    for x_batch, labels_batch in data_loader:
        if num_batches % 1000 == 0:
            print(f"Evaluating batch {num_batches + 1}/{len(data_loader)} for training...") 
        x_batch = x_batch.to(device)
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad()  # Zero the gradients before the backward pass
        # Forward pass
        outputs = model(x_batch)  # (batch_size, output_size (1))
        loss_mse = criterion(outputs, labels_batch)
        with torch.no_grad():
            batch_totals = losses(outputs, labels_batch)
        # Backward pass and optimization
        loss_mse.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        num_batches += 1

        _add_totals(totals, batch_totals)

    return _metrics(totals, "train")

def test_batch(model, criterion, data_loader, device, type="val", per_ticker=False):
    """
    Evaluates the model on a single batch of data.

    Args:
        model (nn.Module): The model to be evaluated.
        criterion (nn.Module): The loss function to compute the loss.
        data_loader (DataLoader): The DataLoader providing batches of evaluation data.
        device (torch.device): The device to run the evaluation on (CPU or GPU).
        type (str): The type of evaluation (e.g., "val", "test").

    Returns:
        float: The computed loss for the current batch.
    """
    num_batches = 0
    offset = 0
    totals = _empty_totals()
    ticker_totals = {}
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for x_batch, labels_batch in data_loader:
            if num_batches % 1000 == 0:
                print(f"Evaluating batch {num_batches + 1}/{len(data_loader)} for {type}...")
            x_batch = x_batch.to(device)
            labels_batch = labels_batch.to(device)
            # Forward pass
            outputs = model(x_batch)
            batch_totals = losses(outputs, labels_batch)

            num_batches += 1
            _add_totals(totals, batch_totals)

            if per_ticker:
                outputs_cpu = outputs.cpu()
                labels_cpu = labels_batch.cpu()
                ticker_ids = data_loader.dataset.ticker_ids[offset:offset + labels_batch.numel()]
                for ticker_id in torch.unique(ticker_ids).tolist():
                    mask = ticker_ids == ticker_id
                    ticker_total = ticker_totals.setdefault(ticker_id, _empty_totals())
                    _add_totals(ticker_total, losses(outputs_cpu[mask], labels_cpu[mask]))
                offset += labels_batch.numel()

    ticker_metrics = []
    for ticker_id, ticker_total in ticker_totals.items():
        row = {
            "ticker": data_loader.dataset.ticker_names[ticker_id],
            "observations": ticker_total["observations"],
        }
        row.update({key.split("/", 1)[1]: value for key, value in _metrics(ticker_total, type).items()})
        ticker_metrics.append(row)

    return _metrics(totals, type), ticker_metrics

def train(model, train_dataset, val_dataset, optimizer, criterion, num_epochs, batch_size, device, name_run):
    """
    Trains the given model using the provided training data and true prices.

    Args:
        model (nn.Module): The model to be trained.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (nn.Module): The loss function to compute the loss.
        num_epochs (int): The number of epochs for training.
        batch_size (int): The size of each training batch.
        device (torch.device): The device to run the training on (CPU or GPU).
        name_run (str): The name of the current training run for logging purposes.
    """
    # Create DataLoaders for training and validation data
    print(f"Creating DataLoaders with batch size: {batch_size}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"DataLoaders created. Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")

    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    os.makedirs("model/checkpoints", exist_ok=True)
    checkpoint_path = f"model/checkpoints/{name_run}_best.pth"

    for epoch in range(num_epochs): 
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        train_metrics = train_batch(model, optimizer, criterion, train_dataloader, device)
        val_metrics, _ = test_batch(model, criterion, val_dataloader, device, type="val")
        epoch_logs = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        wandb.log(epoch_logs)  # Log the epoch logs to wandb

        if (epoch + 1) == 1:
            memory() 

        current_val_loss = val_metrics['val/mse']
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_metrics['train/mse']:.4f}, Val Loss: {val_metrics['val/mse']:.4f}")

        if current_val_loss < best_val_loss:

            best_val_loss = current_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model checkpoint...")
        
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['train/mse'],
                'val_loss': val_metrics['val/mse'],
            }
            print(f"Saving model checkpoint to {checkpoint_path}...")

            torch.save(checkpoint, checkpoint_path)  # Save the model checkpoint
        else:
            print(f"No improvement in validation loss. Current: {current_val_loss:.4f}, Best: {best_val_loss:.4f}")

    wandb.save(checkpoint_path)
    return checkpoint_path

def parameters(model):
    # Total parameters (including frozen/non-trainable)
    total_params = sum(p.numel() for p in model.parameters())

    # Trainable parameters only (requires_grad = True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Rough calculation of parameter size in MB (4 bytes per float32 parameter)

    trainable_params_size = (trainable_params * 4) / (1024 ** 2)
    param_size_mb = (total_params * 4) / (1024 ** 2)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Trainable Parameters Memory: {trainable_params_size:.2f} MB")
    print(f"Model Weights Memory: {param_size_mb:.2f} MB")
    print("-" * 50)

def memory(): 
    if not torch.cuda.is_available():
        return

    allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
    max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Allocated VRAM: {allocated_mb:.2f} MB")
    print(f"Reserved VRAM:  {reserved_mb:.2f} MB")
    print(f"Peak VRAM:      {max_allocated_mb:.2f} MB")

    
        
def load_model(model, checkpoint_path): 
    """
    Loads a model from a checkpoint file.

    Args:
        model (nn.Module): The model architecture to load the weights into.
        checkpoint_path (str): The path to the checkpoint file.
    Returns: 
        model (nn.Module): The model with loaded weights.
        epoch (int): The epoch at which the checkpoint was saved.
        train_loss (float): The training loss at the time of saving.
        val_loss (float): The validation loss at the time of saving.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load to CPU first
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
        return model, checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']
    else:
        print(f"Checkpoint file not found at {checkpoint_path}")
