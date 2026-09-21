import torch
from torch import nn
from torch.nn import functional as F


class HARNet(nn.Module):
    """Hierarchical causal 1/5/20-observation convolutions on nonnegative variance."""

    def __init__(self):
        super().__init__()
        self.week = nn.Conv1d(1, 1, 5, bias=False)
        self.month = nn.Conv1d(1, 1, 4, dilation=5, bias=False)
        self.output = nn.Linear(3, 1)

    def initialize_from_har(self, coefficients):
        coefficients = torch.as_tensor(coefficients, dtype=self.output.weight.dtype, device=self.output.weight.device)
        if coefficients.shape != (4,) or not torch.isfinite(coefficients).all():
            raise ValueError("four finite HAR coefficients are required")
        with torch.no_grad():
            self.week.weight.fill_(1 / 5)
            self.month.weight.fill_(1 / 4)
            self.output.bias.copy_(coefficients[:1])
            self.output.weight.copy_(coefficients[1:].reshape(1, 3))

    def forward(self, x):
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 2 or x.shape[1] < 20:
            raise ValueError("HARNet needs (batch, at least 20) variance history")
        x = x[:, None, :]
        week = F.relu(self.week(x))
        month = F.relu(self.month(week))
        features = torch.cat((x[:, :, -1], week[:, :, -1], month[:, :, -1]), dim=1)
        return self.output(features)
