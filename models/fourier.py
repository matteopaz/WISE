import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pywt

device = "cuda" if torch.cuda.is_available() else "cpu"

    
class CNFourierModel(nn.Module):
    def __init__(self, samples, out, learnsamples=False):
        super().__init__()
        self.samples = samples
        self.out = out
        self.cnn1 = nn.Conv1d(3, 24, 12, padding="same")
        self.cnn2 = nn.Conv1d(24, 48, 7, padding="same")
        self.cnn3 = nn.Conv1d(48, 10, 7, padding="same")
        self.cnn4 = nn.Conv1d(10, 1, 5, padding="same")

        # self.appcnn = lambda x, cnn: cnn(x.permute(0,2,1)).permute(0,2,1)

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//4)
        self.layerthree = nn.Linear(samples//4, self.out)


        self.dp = nn.Dropout(p=0.0)
        
        self.fdft = FDFT(samples, learning=learnsamples, real=True, remove_zerocomp=False)
        
        self.norm = nn.LayerNorm(samples//4)
        self.av = nn.ReLU()

    def forward(self, x):
        N = x.shape[0]
        L = x.shape[1]

        x = x.permute(0,2,1)
        x = self.cnn1(x)
        x = self.av(x)
        x = self.cnn2(x)
        x = self.av(x)
        x = self.cnn3(x)
        x = self.av(x)
        x = self.cnn4(x)
        x = self.av(x).permute(0,2,1)
        x = x.reshape(N,L,1)

        frequential = self.fdft(x).squeeze(-1)
        frequential = torch.arcsinh(frequential)
        # frequential = torch.arcsinh(torch.mul(frequential, snr))

        # plt.scatter(t[0].detach().cpu().numpy(), sig[0].detach().cpu().numpy())
        # frequential = torch.arcsinh(frequential)
        # plt.plot(frequential[0].detach().cpu().numpy())
        # plt.show()
        # raise Exception()

        # frequential = torch.arcsinh(frequential * std)
        
        layone = self.layerone(frequential)
        layone = self.av(layone)
        layone = self.dp(layone)

        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        laytwo = self.dp(laytwo)
        laytwo = self.norm(laytwo)

        laythree = self.layerthree(laytwo)


        return laythree

class WCNFourierModel(nn.Module):
    def __init__(self, samples, out, learnsamples=False):
        super().__init__()
        self.samples = samples
        self.out = out
        self.cnn1 = nn.Conv1d(3, 24, 12, padding="same")
        self.cnn2 = nn.Conv1d(24, 48, 7, padding="same")
        self.cnn3 = nn.Conv1d(48, 10, 7, padding="same")
        self.cnn4 = nn.Conv1d(10, 1, 5, padding="same")

        # self.appcnn = lambda x, cnn: cnn(x.permute(0,2,1)).permute(0,2,1)

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//4)
        self.layerthree = nn.Linear(samples//4, self.out)


        self.dp = nn.Dropout(p=0.0)
        
        self.fdft = FDFT(samples // 2, learning=learnsamples, real=True, remove_zerocomp=False)
        self.wv = Wavelet("db1")
        
        self.norm = nn.LayerNorm(samples//4)
        self.av = nn.ReLU()

    def forward(self, x):
        N = x.shape[0]
        L = x.shape[1]

        x = x.permute(0,2,1)
        x = self.cnn1(x)
        x = self.av(x)
        x = self.cnn2(x)
        x = self.av(x)
        x = self.cnn3(x)
        x = self.av(x)
        x = self.cnn4(x)
        x = self.av(x).permute(0,2,1)
        x = x.reshape(N,L,1)

        frequential = self.fdft(x).squeeze(-1)
        frequential = torch.arcsinh(frequential)
        # frequential = torch.arcsinh(torch.mul(frequential, snr))

        # plt.scatter(t[0].detach().cpu().numpy(), sig[0].detach().cpu().numpy())
        # frequential = torch.arcsinh(frequential)
        # plt.plot(frequential[0].detach().cpu().numpy())
        # plt.show()
        # raise Exception()

        # frequential = torch.arcsinh(frequential * std)
        
        layone = self.layerone(frequential)
        layone = self.av(layone)
        layone = self.dp(layone)

        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        laytwo = self.dp(laytwo)
        laytwo = self.norm(laytwo)

        laythree = self.layerthree(laytwo)


        return laythree
    
class Wavelet(nn.Module):
    def __init__(self, wavelet="db1"):
        super().__init__()
        self.wavelet = wavelet
        self.wave = pywt.Wavelet(wavelet)
    
    def forward(self, x):
        return pywt.wavedec(x, self.wave, mode="periodic")

class FUDFT(nn.Module):
    def __init__(self, samples, learning=False, real=False, remove_zerocomp=True):
        super().__init__()
        self.samples = samples
        self.freqs = torch.linspace(start=0, end=1, steps=samples)
        self.zerocomp = remove_zerocomp

        if learning:
            self.freqs = nn.Parameter(self.freqs)
        
        self.out = lambda x: x
        if real:
            self.out = lambda x: x.abs().to(dtype=torch.float32)
        
        
    def make_fourier(self, N, t):
        freqs = torch.mul(self.freqs, N - 1)
        # exponent_rows = (-2 * torch.pi * 1j * freqs / N).view(1,-1,1) # shape 1,samples,1
        # exponent_cols = (t.unsqueeze(1) * (N - 1)).to(torch.cfloat).to(device) # shape b,1,N
        # exponent = torch.matmul(exponent_rows, exponent_cols)
        exponent_rows = (-2 * torch.pi * 1j * freqs / N)
        exponent_cols = (t * (N - 1)).to(torch.cfloat).to(device)
        exponent = torch.einsum("i, bj -> bij", exponent_rows, exponent_cols)


        fourier = torch.exp(exponent)
        return (1 / np.sqrt(N)) * fourier # of shape (b, samples, N)
        
        
    def forward(self, x: torch.Tensor, t):
        if len(x.shape) != 3:
            raise Exception("Input must be of shape (b, N, C)")
            
        fourier_tens = self.make_fourier(x.size(1), t)
        temp = x.to(torch.cfloat)
        if len(temp.shape) == 2:
            transformed = torch.bmm(fourier_tens, temp)
        else:
            transformed = torch.bmm(fourier_tens, temp)
        if self.zerocomp:
            transformed[:,0] = 0
            transformed[:,-1] = 0
        return self.out(transformed)

class FDFT(nn.Module):
    def __init__(self, samples, learning=False, real=False, remove_zerocomp=True):
        super().__init__()
        self.samples = samples
        self.freqs = torch.linspace(start=0, end=1, steps=samples)
        if learning:
            self.freqs = nn.Parameter(self.freqs)
        
        self.zerocomp = remove_zerocomp
        self.out = lambda x: x
        if real:
            self.out = lambda x: x.abs().to(dtype=torch.float32)
        
        
    def make_fourier(self, N):
        freqs = self.freqs * (N - 1) 
        exponent_rows = (-2 * torch.pi * 1j * freqs / N)
        exponent_cols = torch.arange(N).to(device)
        exponent = torch.outer(exponent_rows, exponent_cols)
        fourier = torch.exp(exponent)
        return (1 / np.sqrt(N)) * fourier # of shape (b, samples, N)
        
        
    def forward(self, x: torch.Tensor, dim=-1):
        if len(x.shape) != 3:
            raise Exception("Input must be of shape (b, N, C)")
        fourier_tens = self.make_fourier(x.size(1)).to(device)
        temp = x.to(torch.cfloat)
        transformed = torch.matmul(fourier_tens, temp)
        if self.zerocomp:
            transformed[:,0] = 0
            transformed[:,-1] = 0
        return self.out(transformed)