import torch
from torch import nn
from torch.nn import functional as F


class HARNet(nn.Module):
    """Hierarchical causal 1/5/20/40/80-observation variance features."""

    def __init__(self):
        super().__init__()
        self.week = nn.Conv1d(1, 1, 5, bias=False)
        self.month = nn.Conv1d(1, 1, 4, dilation=5, bias=False)
        self.forty = nn.Conv1d(1, 1, 2, dilation=20, bias=False)
        self.eighty = nn.Conv1d(1, 1, 2, dilation=40, bias=False)
        self.output = nn.Linear(5, 1)

    def initialize_from_har(self, coefficients):
        coefficients = torch.as_tensor(coefficients, dtype=self.output.weight.dtype, device=self.output.weight.device)
        if coefficients.shape != (6,) or not torch.isfinite(coefficients).all():
            raise ValueError("six finite HAR coefficients are required")
        with torch.no_grad():
            self.week.weight.fill_(1 / 5)
            self.month.weight.fill_(1 / 4)
            self.forty.weight.fill_(1 / 2)
            self.eighty.weight.fill_(1 / 2)
            self.output.bias.copy_(coefficients[:1])
            self.output.weight.copy_(coefficients[1:].reshape(1, 5))

    def forward(self, x):
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 2 or x.shape[1] < 80:
            raise ValueError("HARNet-80 needs (batch, at least 80) variance history")
        x = x[:, None, :]
        week = F.relu(self.week(x))
        month = F.relu(self.month(week))
        forty = F.relu(self.forty(month))
        eighty = F.relu(self.eighty(forty))
        features = torch.cat((x[:, :, -1], week[:, :, -1], month[:, :, -1],
                              forty[:, :, -1], eighty[:, :, -1]), dim=1)
        return self.output(features)


def train(args):
    from models.variance_neural import neural_train
    return neural_train('harnet_80', args)


def predict(fit, frame, split='test'):
    from models.variance_neural import neural_predict
    return neural_predict('harnet_80', fit, frame, split)


if __name__ == '__main__':
    from models.variance_neural import main
    main('harnet_80')
