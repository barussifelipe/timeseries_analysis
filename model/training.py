from model import *
import wandb

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

    for epoch in range(num_epochs):
        running_loss = 0.0 
        model.train() 
        for batch_idx in range(0, len(train_data), batch_size):  # Assuming batch size of 32
            batched_train_data = train_data[batch_idx:batch_idx + batch_size]
            batched_train_labels = train_labels[batch_idx:batch_idx + batch_size]

            optimizer.zero_grad()

            # Forward pass

            #Train pass
            outputs = model(batched_train_data)
            loss = criterion(outputs, batched_train_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        wandb.log({"train_loss": loss.item(), "val_loss": loss_val.item()})

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {loss_val.item():.4f}')