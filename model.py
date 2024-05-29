import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 128):
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state, t):
        x = F.relu(self.fc1(state))
        x = F.softmax(self.fc2(x), dim=-1)
        return x
    

class BasicRNNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 128):
        super(BasicRNNNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(state_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, action_size)
        self.h_prev = None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / self.hidden_size ** 0.5
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, state, t):
        if t == 0:
            self.h_prev = self.init_hidden(batch_size = 1)
        h_t = self.W_xh(state) + self.W_hh(self.h_prev)
        h_t = torch.tanh(h_t)
        x = F.softmax( self.W_hy(h_t), dim= -1 )
        self.h_prev = h_t.detach()
        return x

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size)
    
class LSTMNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 128):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(state_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x, t):
        if t == 0:
            self.h_prev = self.init_hidden(batch_size = 1)
        x = x.reshape(1,1,-1)
        x, self.h_prev = self.lstm(x, self.h_prev)
        self.h_prev = (self.h_prev[0].detach(), self.h_prev[1].detach())

        out = x[:, -1, :]
        x = F.softmax(self.fc(out), dim=-1)
        return x

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(1, batch_size, self.hidden_size)
        c0 = torch.zeros(1, batch_size, self.hidden_size)
        return (h0, c0)