import torch.nn as nn
import torch

class LSTMModel(nn.Module):
      # input_size : number of features in input at each time step
      # hidden_size : Number of LSTM units
      # num_layers : number of LSTM layers
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTMModel, self).__init__() #initializes the parent class nn.Module
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
   
        self.fc1 = nn.Linear(hidden_size, output_size)



    def forward(self, x): # defines forward pass of the neural network
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        out, _ = self.lstm1(x, (h0, c0))

        out = out[:, -1, :]
        out = self.fc1(out)
        # print(out.shape)
        return out.unsqueeze(-1)