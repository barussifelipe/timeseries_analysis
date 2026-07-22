from model import *
import wandb
import torch 
from torch.utils.data import DataLoader, TensorDataset
import os 




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
    model.to(device)
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    val_data = val_data.to(device)
    val_labels = val_labels.to(device)

    # Create DataLoaders for training and validation data
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs): 
        train_metrics = train_batch(model, optimizer, criterion, train_dataloader)
        val_metrics = test_batch(model, criterion, val_dataloader, type="val")
        epoch_logs = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        wandb.log(epoch_logs)  # Log the epoch logs to wandb

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_metrics['train/mse']:.4f}, Val Loss: {val_metrics['val/mse']:.4f}")
        
        checkpoint_path = f"model/checkpoints/{name_run}_epoch_{epoch + 1}.pth"

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['train/mse'],
            'val_loss': val_metrics['val/mse'],
        }

        torch.save(checkpoint, checkpoint_path)  # Save the model checkpoint
        wandb.save(checkpoint_path)  # Save the checkpoint to wandb

    return train_metrics, val_metrics  # Return the final training and validation metrics

def parameters_memory(model):
    # Total parameters (including frozen/non-trainable)
    total_params = sum(p.numel() for p in model.parameters())

    # Trainable parameters only (requires_grad = True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if torch.cuda.is_available():
        # 1. Memory currently occupied by tensors/weights (in Bytes -> MB)
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        
        # 2. Total memory reserved by PyTorch's caching allocator
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        
        # 3. Peak memory used during the run (great for finding batch size limits)
        max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Allocated VRAM: {allocated_mb:.2f} MB")
    print(f"Reserved VRAM:  {reserved_mb:.2f} MB")
    print(f"Peak VRAM:      {max_allocated_mb:.2f} MB")

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
        
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