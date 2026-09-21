from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(input_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.network(x)


def train(args):
    from models.variance_neural import neural_train
    return neural_train('mlp', args)

def predict(fit, frame, split='test'):
    from models.variance_neural import neural_predict
    return neural_predict('mlp', fit, frame, split)

if __name__ == '__main__':
    from models.variance_neural import main
    main('mlp')
