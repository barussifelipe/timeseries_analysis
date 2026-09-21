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
    def forward(self, x): 
        """
        x shape: (batch_size, number_of_days, number_of_features)
        """
        batch_size, number_of_days, _ = x.shape
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device) 
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(number_of_days):
            x_t = x[:, t, :]  # (batch_size, input_size)
            h_t, c_t = self.cell(x_t, h_t, c_t)
        
        output = self.output_layer(h_t)  # (batch_size, output_size)

        return output


def fit_variance_network(model, train_data, val_data, floor, path, settings,
                         epochs=20, batch_size=128, patience=5, run=None):
    """Mean window QLIKE, clipped gradients, one learning-rate retry."""
    from torch.utils.data import DataLoader
    from models.training_blocks import floor_prediction, qlike, save_fit

    if not len(train_data) or not len(val_data):
        raise ValueError('neural fitting needs train and validation windows')
    if patience < 1:
        raise ValueError('patience must be positive')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    best, stale, retried = float('inf'), 0, False
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = qlike(y, floor_prediction(model(x), floor))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = sum(float(qlike(y, floor_prediction(model(x), floor))) * len(y)
                        for x, y in val_loader) / len(val_data)
        if run:
            run.log({'epoch': epoch, 'val/qlike': score, 'learning_rate': optimizer.param_groups[0]['lr']})
        if score < best:
            best, stale = score, 0
            save_fit(path, {'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch, 'val_qlike': score, 'floor': floor,
                            'settings': settings})
        else:
            stale += 1
            if stale >= patience:
                if retried:
                    break
                optimizer.param_groups[0]['lr'] *= .1
                retried, stale = True, 0
    if best == float('inf'):
        raise RuntimeError('no finite validation checkpoint')
    return path


def train(args):
    from models.variance_neural import neural_train
    return neural_train('base_lstm_vol', args)


def predict(fit, frame, split='test'):
    from models.variance_neural import neural_predict
    return neural_predict('base_lstm_vol', fit, frame, split)


if __name__ == '__main__':
    from models.variance_neural import main
    main('base_lstm_vol')
    

