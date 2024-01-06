import torch
import torch.nn as nn
import numpy as np
import time

dropout = 0.5

class AnomalyDetector(nn.Module):
    def __init__(self, emb_dim=2, out=3):
        super().__init__()
        self.dp = nn.Dropout(p=dropout)
        self.sequence_embedding_size = 6
        
        self.blend_bilinear = nn.Bilinear(self.sequence_embedding_size, self.sequence_embedding_size, self.sequence_embedding_size)

        self.selfattn_signal = nn.MultiheadAttention(2, 2, dropout=dropout, batch_first=True)
        self.selfattn_fourier = nn.MultiheadAttention(2, 2, dropout=dropout, batch_first=True)
        
        self.to_out = nn.Linear(self.sequence_embedding_size, out)

    def forward(self, x):
        # input shape is Batches x Time x Features
        N = x.shape[0]
        T = x.shape[1]
        fourier = torch.fft.fft(x, dim=1) 
        # assuming x is (Light, Standard Deviation, Portion Elapsed)
        std = torch.max(x[:, :, 1], dim=1)[0] # Gives N-long vector 
        x = x[:, :, (0,2)] # Removed std feature
        
        selfattend_signal = self.selfattn_signal(x, x, x)[0]
        selfattend_fourier = self.selfattn_fourier(x,x,x)[0]
        
        
        postattn_signal = torch.prod(selfattend_signal, dim=2).squeeze() # Multiplies dim=2 (embed_dim) with itself down into 1 row (N x L)
        postattn_signal = self.finite_encoding(postattn_signal, dim=1, to=self.sequence_embedding_size)
        postattn_fourier = torch.prod(selfattend_fourier, dim=2).squeeze() # Multiplies dim=2 (embed_dim) with itself down into 1 row (N x L)
        postattn_fourier = self.finite_encoding(postattn_fourier, dim=1, to=self.sequence_embedding_size)
        # maybe add FC interlayer here
        blend = self.blend_bilinear(postattn_signal, postattn_fourier)
        out = self.to_out(blend)
        return out

        
    
    def finite_encoding(self, inp, dim, to):
        lastdim = len(inp.shape) - 1
        indexes = torch.randperm(inp.shape[dim])[inp.shape[dim]% to:].sort()[0]
        inp = torch.index_select(inp, dim, indexes)
        out = torch.transpose(nn.MaxPool1d(inp.shape[dim] // to)(torch.transpose(inp, dim, lastdim)), dim, lastdim)
        return out

    
inst = AnomalyDetector()
test = torch.arange(30).reshape(1,10,3).to(torch.float)
print(test)
test = torch.prod(test, dim=2)
print(test)
test = inst.finite_encoding(test, dim=1, to=5)
print(test)