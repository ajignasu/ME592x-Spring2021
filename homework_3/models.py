import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Fully_Connected(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Fully_Connected, self).__init__()
        
        self.layer_1 = nn.Linear(in_dims, 8*out_dims)
        self.layer_2 = nn.Linear(8*out_dims, 4*out_dims)
        self.layer_3 = nn.Linear(4*out_dims, 2*out_dims)
        self.layer_4 = nn.Linear(2*out_dims, out_dims)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        return self.layer_4(x)


# Recurrent neural network (many-to-one)
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py#L39-L58
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out