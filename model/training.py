from model import *
import wandb
import torch 
from torch.utils.data import DataLoader, TensorDataset

wandb.init(
    entity="personalfeb",  # Name of your project
    name="experiment_hidden64_lr001",   # Optional: specific run name
    config={                            # Store model hyperparameters
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "hidden_size": 64,
        "window_size": 30
    }
)

def losses(outputs, labels, epsilon=1e-8):
    """
    Computes the Mean Squared Error (MSE) loss between the model's outputs and the true labels.

    Args:
        outputs (torch.Tensor): The model's predicted outputs.
        labels (torch.Tensor): The true labels corresponding to the inputs.

    Returns:
        loss_mae (torch.Tensor): The computed Mean Absolute Error (MAE) loss.
        loss_mse (torch.Tensor): The computed Mean Squared Error (MSE) loss.
        loss_rmse (torch.Tensor): The computed Root Mean Squared Error (RMSE) loss.
        loss_mape (torch.Tensor): The computed Mean Absolute Percentage Error (MAPE) loss.
    """
    loss_mae = torch.mean(torch.abs(outputs - labels))  # Mean Absolute Error
    loss_rmse = torch.sqrt(torch.mean((outputs - labels) ** 2) )                     # Root Mean Squared Error
    loss_mape = torch.mean(torch.abs((labels - outputs) / (labels + epsilon))) * 100  # Mean Absolute Percentage Error
    loss_r2 = 1 - (torch.sum((labels - outputs) ** 2) / torch.sum((labels - torch.mean(labels)) ** 2))  # R-squared

    return loss_mae, loss_rmse, loss_mape, loss_r2

def train_batch(model, optimizer, criterion, data_loader):
    """
    Trains the model on a single batch of data.

    Args:
        model (nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (nn.Module): The loss function to compute the loss.
        data_loader (DataLoader): The DataLoader providing batches of training data.

    Returns:
        float: The computed loss for the current batch.
    """
    num_batches = 0 
    total_loss_mse = 0.0
    total_loss_rmse = 0.0
    total_loss_mape = 0.0
    total_loss_mae = 0.0
    total_loss_r2 = 0.0

    model.train()  # Set the model to training mode
    for x_batch, labels_batch in data_loader:
        optimizer.zero_grad()  # Zero the gradients before the backward pass
        # Forward pass
        outputs = model(x_batch)
        loss_mse = criterion(outputs, labels_batch)
        with torch.no_grad():  # Disable gradient computation for loss metrics
            loss_mae, loss_rmse, loss_mape, loss_r2 = losses(outputs, labels_batch)
        # Backward pass and optimization
        loss_mse.backward()
        optimizer.step()

        num_batches += 1

        total_loss_mse += loss_mse.item()
        total_loss_rmse += loss_rmse.item()
        total_loss_mape += loss_mape.item()
        total_loss_mae += loss_mae.item()
        total_loss_r2 += loss_r2.item()

    avg_loss_mse = total_loss_mse / num_batches if num_batches > 0 else 0.0
    avg_loss_rmse = total_loss_rmse / num_batches if num_batches > 0 else 0.0
    avg_loss_mape = total_loss_mape / num_batches if num_batches > 0 else 0.0
    avg_loss_mae = total_loss_mae / num_batches if num_batches > 0 else 0.0
    avg_loss_r2 = total_loss_r2 / num_batches if num_batches > 0 else 0.0

    return {
        "train/mse": avg_loss_mse,
        "train/rmse": avg_loss_rmse,
        "train/mape": avg_loss_mape,
        "train/mae": avg_loss_mae,
        "train/r2": avg_loss_r2,
    }  # Return the average loss values for logging

def test_batch(model, criterion, data_loader, type="val"):
    """
    Evaluates the model on a single batch of data.

    Args:
        model (nn.Module): The model to be evaluated.
        criterion (nn.Module): The loss function to compute the loss.
        data_loader (DataLoader): The DataLoader providing batches of evaluation data.

    Returns:
        float: The computed loss for the current batch.
    """
    num_batches = 0 
    total_loss_mse = 0.0
    total_loss_rmse = 0.0
    total_loss_mape = 0.0
    total_loss_mae = 0.0
    total_loss_r2 = 0.0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for x_batch, labels_batch in data_loader:
            # Forward pass
            outputs = model(x_batch)
            loss_mse = criterion(outputs, labels_batch)
            loss_mae, loss_rmse, loss_mape, loss_r2 = losses(outputs, labels_batch)

            num_batches += 1

            total_loss_mse += loss_mse.item()
            total_loss_rmse += loss_rmse.item()
            total_loss_mape += loss_mape.item()
            total_loss_mae += loss_mae.item()
            total_loss_r2 += loss_r2.item()

        avg_loss_mse = total_loss_mse / num_batches if num_batches > 0 else 0.0
        avg_loss_rmse = total_loss_rmse / num_batches if num_batches > 0 else 0.0
        avg_loss_mape = total_loss_mape / num_batches if num_batches > 0 else 0.0
        avg_loss_mae = total_loss_mae / num_batches if num_batches > 0 else 0.0
        avg_loss_r2 = total_loss_r2 / num_batches if num_batches > 0 else 0.0

    return {
        f"{type}/mse": avg_loss_mse,
        f"{type}/rmse": avg_loss_rmse,
        f"{type}/mape": avg_loss_mape,
        f"{type}/mae": avg_loss_mae,
        f"{type}/r2": avg_loss_r2,
    }  # Return the average loss values for logging

def train(model, train_data, train_labels, val_data, val_labels, optimizer, criterion, num_epochs, batch_size, device):
    """
    Trains the given model using the provided training data and true prices.

    Args:
        model (nn.Module): The model to be trained.
        train_data (torch.Tensor): The input training data.
        train_labels (torch.Tensor): The true labels corresponding to the training data.
        val_data (torch.Tensor): The input validation data.
        val_labels (torch.Tensor): The true labels corresponding to the validation data.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (nn.Module): The loss function to compute the loss.
        num_epochs (int): The number of epochs for training.
        batch_size (int): The size of each training batch.
        device (torch.device): The device to run the training on (CPU or GPU).
    """
    model.to(device)
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    val_data = val_data.to(device)
    val_labels = val_labels.to(device)

    # Create DataLoaders for training and validation data
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs): 
        train_metrics = train_batch(model, optimizer, criterion, train_dataloader)
        val_metrics = test_batch(model, criterion, val_dataloader, type="val")
        epoch_logs = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        wandb.log(epoch_logs)  # Log the epoch logs to wandb

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_metrics['train/mse']:.4f}, Val Loss: {val_metrics['val/mse']:.4f}")