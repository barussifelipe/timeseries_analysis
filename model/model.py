import torch.nn as nn 
import torch.nn.init as init 
import torch 

class FEBCellLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size 

        concat_size = input_size + hidden_size

        #The weights have the size of the hidden state, since the objective is the hidden state. 
        self.input_gate = nn.Linear(concat_size, hidden_size, bias = True) #Input Gate, wheter 
        self.forget_gate = nn.Linear(concat_size, hidden_size, bias = True) #Forget Gate 
        self.candidate_gate = nn.Linear(concat_size, hidden_size, bias = True) #Candidate Gate 
        self.output_gate = nn.Linear(concat_size, hidden_size, bias = True) #Output Gate 
        
        #Xavier initialization is useful for symmetric activation functions. 
        init.xavier_uniform_(self.input_gate.weight)  
        init.xavier_uniform_(self.forget_gate.weight)
        init.xavier_uniform_(self.candidate_gate.weight)
        init.xavier_uniform_(self.output_gate.weight)

        init.zeros_(self.input_gate.bias)
        init.zeros_(self.candidate_gate.bias)
        init.zeros_(self.output_gate.bias)

        #sigmoid(1) = 0.731, so the forget gate will be more likely to remember the previous cell state
        init.ones_(self.forget_gate.bias) 

    def forward(self, x_t, h_prev, c_prev): 
        #Concatenate the input and the previous hidden state. z = xW^T + h_prevU^T + b. W^T and U^T are the weights of the input and hidden state respectively. We can concatennate the input and the previous hidden state since they are both multiplied by their respective weights and summed in the end. 
        """
        x_t:    (batch_size, input_size)  - Input for the current time step
        h_prev: (batch_size, hidden_size) - Previous hidden state
        c_prev: (batch_size, hidden_size) - Previous cell state
        """
        concat = torch.cat((h_prev, x_t), dim=1) 

        #Compute the gates
        i_t = torch.sigmoid(self.input_gate(concat)) 
        f_t = torch.sigmoid(self.forget_gate(concat)) 
        g_t = torch.tanh(self.candidate_gate(concat)) 
        o_t = torch.sigmoid(self.output_gate(concat)) 

        #Compute the new cell state and hidden state
        c_t = f_t * c_prev + i_t * g_t 
        h_t = o_t * torch.tanh(c_t) 

        return h_t, c_t
    
class FEBLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int): 
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.cell = FEBCellLSTM(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size, bias=True)
        #Since the output of the LSTM is a value between -1 and 1, we can use a Xavier initialization for the output layer.
        init.xavier_uniform_(self.output_layer.weight)
        init.zeros_(self.output_layer.bias)
    def forward(self, x, h_prev, c_prev): 
        """
        x shape: (batch_size, sequence_length(window_size), input_size) -> e.g (32, 30, 1) 
        """
        batch_size, sequence_length, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device) 
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(sequence_length):
            x_t = x[:, t, :]  # (batch_size, input_size)
            h_t, c_t = self.cell(x_t, h_t, c_t)
        
        output = self.output_layer(h_t)  # (batch_size, output_size)

        return output, h_t, c_t
    

