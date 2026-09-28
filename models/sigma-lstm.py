"""Sigma-LSTM adapted to forecast next-observation GK log volatility.

Input is z-scored GK log volatility, shaped (batch, time, 1). Outputs are the next
log-volatility estimate and squared mean cell state, each shaped (batch, 1).
The latter is treated as log-volatility variance, not a GK variance forecast.
Equation (8)'s gate variance is softplus(W_o[C(t)^2]); tanh is used for its
unspecified phi in Equation (9). These are project choices.
The stochastic gate is sampled in both training and evaluation modes.
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.base_lstm import FEBCellLSTM, FEBLSTM


class SigmaLSTMCell(FEBCellLSTM):
    def __init__(self, hidden_size: int):
        if hidden_size < 1:
            raise ValueError('hidden_size must be positive')
        super().__init__(1, hidden_size)
        self.output_gate = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, standardized_log_volatility, hidden, memory):
        joined = torch.cat((hidden, standardized_log_volatility), dim=1)

        f_t = torch.sigmoid(self.forget_gate(joined))
        i_t = torch.sigmoid(self.input_gate(joined))
        g_t = torch.tanh(self.candidate_gate(joined))

        memory_t = (f_t * memory + i_t * g_t)
        gate_variance = F.softplus(self.output_gate(memory_t.square()))
        gate = torch.randn_like(gate_variance) * torch.sqrt(
            gate_variance + torch.finfo(gate_variance.dtype).tiny)
        return gate * torch.tanh(memory_t), memory_t


class SigmaLSTM(FEBLSTM):
    def __init__(self, hidden_size: int):
        super().__init__(1, hidden_size, 1)
        self.cell = SigmaLSTMCell(hidden_size)
        self.capture_memory = False
        # The bias lets the log-volatility head represent a negative base level.
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, standardized_log_volatility):
        if standardized_log_volatility.ndim != 3 or standardized_log_volatility.shape[-1] != 1 or standardized_log_volatility.shape[1] == 0:
            raise ValueError('standardized_log_volatility must have shape (batch, time, 1) with time > 0')
        hidden = standardized_log_volatility.new_zeros((standardized_log_volatility.shape[0], self.hidden_size))
        memory = torch.ones_like(hidden)
        memory_trace = [] if self.capture_memory else None
        for time in range(standardized_log_volatility.shape[1]):
            hidden, memory = self.cell(standardized_log_volatility[:, time], hidden, memory)
            if memory_trace is not None:
                memory_trace.append(memory)
        result = self.output_layer(hidden), memory.mean(dim=-1, keepdim=True).square()
        return (*result, torch.stack(memory_trace)) if memory_trace is not None else result


if __name__ == '__main__':
    from models.variance_neural import main
    main('sigma_lstm')
