import torch
from torch.nn import functional as F

from models.base_lstm import FEBCellLSTM, FEBLSTM


class SiLUCellLSTM(FEBCellLSTM):
    def forward(self, x_t, h_prev, c_prev):
        concat = torch.cat((h_prev, x_t), dim=1)
        i_t = torch.sigmoid(self.input_gate(concat))
        f_t = torch.sigmoid(self.forget_gate(concat))
        g_t = F.silu(self.candidate_gate(concat))
        o_t = torch.sigmoid(self.output_gate(concat))
        c_t = f_t * c_prev + i_t * g_t
        return o_t * F.silu(c_t), c_t


class SiLULSTM(FEBLSTM):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__(input_size, hidden_size, output_size)
        self.cell = SiLUCellLSTM(input_size, hidden_size)


def train(args):
    from models.variance_neural import neural_train
    return neural_train('silu_lstm', args)

def predict(fit, frame, split='test'):
    from models.variance_neural import neural_predict
    return neural_predict('silu_lstm', fit, frame, split)

if __name__ == '__main__':
    from models.variance_neural import main
    main('silu_lstm')
