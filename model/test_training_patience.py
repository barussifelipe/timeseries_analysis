from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

import training


def test_learning_rate_retry_and_early_stop():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataset = TensorDataset(torch.zeros(1, 1), torch.zeros(1, 1))
    validation_losses = iter((1.0, 1.1, 1.1, 1.1, 1.1))
    logs = []
    final_path = Path("model/checkpoints/patience_check_best_epoch1.pth")
    temporary_path = Path("model/checkpoints/patience_check_best.pth")

    try:
        with (
            patch.object(training, "train_batch", return_value={"train/mse": 1.0}),
            patch.object(
                training, "test_batch",
                side_effect=lambda *args, **kwargs: ({"val/mse": next(validation_losses)}, []),
            ),
            patch.object(training, "memory"),
            patch.object(training.wandb, "log", side_effect=logs.append),
            patch.object(training.wandb, "save"),
        ):
            checkpoint_path = training.train(
                model, dataset, dataset, optimizer, torch.nn.MSELoss(),
                num_epochs=20, batch_size=1, device=torch.device("cpu"),
                name_run="patience_check", patience=2,
            )

        assert Path(checkpoint_path) == final_path
        assert len(logs) == 5
        assert logs[2]["learning_rate"] == 0.1
        assert abs(logs[3]["learning_rate"] - 0.01) < 1e-12
        assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-12
    finally:
        final_path.unlink(missing_ok=True)
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_learning_rate_retry_and_early_stop()
    print("learning-rate retry check passed")
