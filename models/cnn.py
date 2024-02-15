import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

class CNN(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.av = nn.ReLU()
        self.dp = nn.Dropout(p=0.3)

        self.conv1 = nn.Conv1d(3, 12, 4, padding="same")
        self.conv2 = nn.Conv1d(12, 32, 8, padding="same")
        self.conv3 = nn.Conv1d(32, 64, 8, padding="same")
        self.conv4 = nn.Conv1d(64, 32, 8, padding="same")
        self.conv5 = nn.Conv1d(32, 16, 8, padding="same")
        
        self.pool = nn.MaxPool1d(2)

        self.prelstm = lambda x: torch.transpose(x, 1, 2)
        self.lstm = nn.LSTM(16, 24, 2, batch_first=True, dropout=0.5)
        self.getlstmout = lambda x: x[0][:, -1, :]


        self.fc1 = nn.Linear(24, 8)
        self.fc2 = nn.Linear(8, out)



    def forward(self, xt):
        x = torch.transpose(xt, 1, 2) # to N x C x L
        avdp = lambda x: self.av(self.dp(x))

        x = avdp(self.conv1(x))
        x = avdp(self.conv2(x))
        x = avdp(self.conv3(x))
        x = avdp(self.conv4(x))
        x = self.pool(x)
        x = avdp(self.conv5(x))
        x = self.pool(x)
        
        x = self.prelstm(x)
        x = self.getlstmout(self.lstm(x))
        x = avdp(x)
        x = avdp(self.fc1(x))
        x = self.fc2(x)
        return x

        
    def finite_encoding(self, inp, to, dim):
        indexes = torch.randperm(inp.shape[dim])[inp.shape[dim]% to:].sort()[0]
        inp = torch.index_select(inp, dim, indexes)
        out = nn.MaxPool1d(inp.shape[dim] // to)(inp)
        return out