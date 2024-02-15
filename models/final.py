import torch
import torch.nn as nn
import numpy as np
from fourier import FDFT

device = "cuda" if torch.cuda.is_available() else "cpu"

dropout = 0.4



class AnomalyDetector(nn.Module):
    def __init__(self, emb_dim=16, out=3):
        super().__init__()
        self.outdim = out
        self.dp = nn.Dropout(p=dropout)
        self.av = nn.ReLU()
        
        # self.transform_to_QKV = nn.LSTM(3, 3 * emb_dim, batch_first=True, num_layers=n_layers, dropout=dropout)
        initial_features = 5
        
        self.emb_1 = nn.Linear(initial_features, 2*emb_dim)
        self.emb_2 = nn.Linear(2*emb_dim, 3*emb_dim)
        
        # self.attn = nn.MultiheadAttention(emb_dim, num_heads=emb_dim//4, dropout=dropout)
        self.attn = RepeatedAttention(emb_dim, 2, dropout=dropout, activation=self.av, fc=True)
        
        hidden = emb_dim * 4
        
        self.decode = FDFT(samples=hidden, learning=False, real=True)
        self.linout_1 = nn.Linear(hidden, hidden//2)
        self.linout_2 = nn.Linear(hidden//2, out)
        self.de_emb = nn.Linear(emb_dim, 1)
        
        # Lots of information loss in the AvgPool / finite seqembed step. Maybe use a RNN instead or a different method that has less loss
        


    def forward(self, x: torch.Tensor):
        # input shape is Batches x Time x Features
        N = x.shape[0]
        L = x.shape[1]
        # assuming x is (Light, Standard Deviation, Portion Elapsed)
        std = x[:, :, 1].reshape(N,L) # Gives N-long vector 
        t = x[:,:,2].reshape(N,L) # N x L 
        x = x[:, :, 0].reshape(N,L) # N x L 
        f1 = torch.fft.fft(x, dim=1)
        f1 = torch.abs(f1).reshape(N,L) # N x L 
        f2 = torch.fft.fft(f1, dim=1)
        f2 = torch.abs(f1).reshape(N,L)
        
        datatens = torch.stack((x,f1,f2,t,std), dim=2)
        
        emb = self.dp(self.av(self.emb_2(self.dp(self.av(self.emb_1(datatens)))))) # Embedding ANN
        
        Q, K, V = torch.split(emb, emb.shape[2] // 3, dim=2)
    
        attn = self.attn(Q,K,V) # N x L x emb_dim
        
        decoded = self.decode(attn, dim=-2) # Fourier
        de_emb = self.dp(self.av(self.de_emb(decoded))).squeeze()
        out1 = self.dp(self.av(self.linout_1(de_emb)))
        out2 = self.linout_2(out1)
        return out2
    
    
    def finite_encoding(self, inp, dim, to):
        lastdim = len(inp.shape) - 1
        indexes = torch.randperm(inp.shape[dim])[inp.shape[dim]% to:].sort()[0]
        inp = torch.index_select(inp, dim, indexes)
        out = torch.transpose(nn.AvgPool1d(inp.shape[dim] // to)(torch.transpose(inp, dim, lastdim)), dim, lastdim)
        return out



class RepeatedAttention(nn.Module):
    def __init__(self, emb_dim, n, dropout =0.0, activation = nn.ReLU, fc=False):
        super().__init__()
        self.attn = lambda q,k,v: nn.functional.scaled_dot_product_attention(q,k,v, dropout_p=dropout)
        
        self.interlayers = nn.ModuleList([nn.Linear(emb_dim, 3*emb_dim) for _ in range(n - 1)])
        self.av = activation
        self.dp = nn.Dropout(dropout)
        self.split = lambda t: torch.split(t, t.shape[-1] // 3, dim=-1)
        self.norm = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(n-1)])
        
        self.fc = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(n-1)])
        self.isfc = fc
        # self.norm = [nn.Identity() for _ in range(n-1)]

        
    def forward(self, Q, K, V):
        current = self.av(self.attn(Q,K,V))
        for i, inter in enumerate(self.interlayers):
            Q, K, V = self.split(self.dp(self.av(inter(current)))) # emb -> 3xemb -dp, av-> Q,K,V
            current = self.norm[i](self.attn(Q,K,V) + current) # TWEAK RESIDUAL HERE
            current = self.av(current)
            if self.isfc:
                current = self.dp(self.av(self.fc[i](current)))
            
            
        return current
            
            
            

# class AnomalyDetector(nn.Module):
#     def __init__(self, emb_dim=4, out=3):
#         super().__init__()
#         self.dp = nn.Dropout(p=dropout)
#         self.av = nn.ReLU()
#         self.sequence_embedding_size = 6
        
#         initial_features = 2
        
#         self.signal_Q = nn.Linear(initial_features, emb_dim)
#         self.signal_K = nn.Linear(initial_features, emb_dim)
#         self.signal_V = nn.Linear(initial_features, emb_dim)
        
#         self.fourier_Q = nn.Linear(initial_features, emb_dim)
#         self.fourier_K = nn.Linear(initial_features, emb_dim)
#         self.fourier_V = nn.Linear(initial_features, emb_dim)
        
#         self.blend_bilinear = nn.Bilinear(emb_dim, emb_dim, 8)

#         self.selfattn = lambda q,k,v: nn.functional.scaled_dot_product_attention(q,k,v)
        
#         self.decode = nn.Linear(8, 1)
        
#         self.to_out = nn.Linear(self.sequence_embedding_size, out)

#     def forward(self, x):
#         # input shape is Batches x Time x Features
#         N = x.shape[0]
#         T = x.shape[1]
#         # assuming x is (Light, Standard Deviation, Portion Elapsed)
#         std = torch.max(x[:, :, 1], dim=1)[0] # Gives N-long vector 
#         x = x[:, :, (0,2)] # Removed std feature
#         fourier = torch.fft.fft(x, dim=1)
#         fourier = torch.abs(fourier)
        
#         sQ = self.av(self.signal_Q(x))
#         sK = self.av(self.signal_K(x))
#         sV = self.av(self.signal_V(x))
#         fQ = self.av(self.fourier_Q(x))
#         fK = self.av(self.fourier_K(x))
#         fV = self.av(self.fourier_V(x))
        
#         selfattend_signal = self.selfattn(sQ, sK, sV)
#         selfattend_fourier = self.selfattn(fQ, fK, fV)
        
#         selfattend_signal = self.dp(self.av(selfattend_signal))
#         selfattend_fourier = self.dp(self.av(selfattend_fourier))
        
#         blend = self.blend_bilinear(selfattend_signal, selfattend_fourier)
#         blend = self.dp(self.av(blend))
        
#         decoded = self.decode(blend).squeeze()
#         decoded = self.dp(self.av(decoded))
        
#         finite = self.finite_encoding(decoded, dim=1, to=self.sequence_embedding_size)
#         # maybe add FC interlayer here
#         # blend = self.dp(self.av(blend))
#         out = self.to_out(finite)
#         return out

        
    
#     def finite_encoding(self, inp, dim, to):
#         lastdim = len(inp.shape) - 1
#         indexes = torch.randperm(inp.shape[dim])[inp.shape[dim]% to:].sort()[0]
#         inp = torch.index_select(inp, dim, indexes)
#         out = torch.transpose(nn.AvgPool1d(inp.shape[dim] // to)(torch.transpose(inp, dim, lastdim)), dim, lastdim)
#         return out