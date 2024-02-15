import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class FourierModel(nn.Module):
    def __init__(self, samples, out, learnsamples=False):
        super().__init__()
        self.samples = samples
        self.out = out
        # self.cnn1 = nn.Conv1d(3, 1, 2, padding="same")
        # self.cnn2 = nn.Conv1d(1, 24, 2, padding="same")
        # self.cnn3 = nn.Conv1d(24, 1, 2, padding="same")

        # self.appcnn = lambda x, cnn: cnn(x.permute(0,2,1)).permute(0,2,1)

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//4)
        
        self.layerthree = nn.Linear(samples//4,self.out)
        self.dp = nn.Dropout(p=0.85)
        
        
        self.fdft = FDFT(samples, learning=learnsamples, real=True, remove_zerocomp=False)
        
        self.norm = nn.LayerNorm(samples//4)
        self.av = nn.ReLU()

    def forward(self, x):
        N = x.shape[0]
        L = x.shape[1]

        sig = x[:,:,0].reshape(N,L,1)
        t = x[:,:,2].reshape(N,L,1)

        temb = torch.mean(t, dim=1).reshape(N,1)
        snr = x[:,:,1].reshape(N,L)
        snr[snr == 0] = torch.nan
        snr = torch.nanmean(snr, dim=-1).reshape(N,1)

        frequential = self.fdft(sig, dim=1).squeeze(-1)
        frequential = torch.arcsinh(torch.mul(frequential, snr))

        # plt.scatter(t[0].detach().cpu().numpy(), sig[0].detach().cpu().numpy())
        # frequential = torch.arcsinh(frequential)
        # plt.plot(frequential[0].detach().cpu().numpy())
        # plt.show()
        # raise Exception()

        # frequential = torch.arcsinh(frequential * std)
        
        layone = self.layerone(frequential)
        # layone = self.norm(layone)
        layone = self.av(layone)
        layone = self.dp(layone)
        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        inter = self.dp(laytwo)
        inter = self.norm(inter)
        
        final = self.layerthree(inter)
        return final
    
class CNFourierModel(nn.Module):
    def __init__(self, samples, out, learnsamples=False):
        super().__init__()
        self.samples = samples
        self.out = out
        self.cnn1 = nn.Conv1d(3, 12, 2, padding="same")
        self.cnn2 = nn.Conv1d(12, 24, 2, padding="same")
        self.cnn3 = nn.Conv1d(24, 1, 2, padding="same")

        # self.appcnn = lambda x, cnn: cnn(x.permute(0,2,1)).permute(0,2,1)

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//4)
        
        self.layerthree = nn.Linear(samples//4,self.out)
        self.dp = nn.Dropout(p=0.85)
        self.lightdp = nn.Dropout(p=0.2)
        
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
        x = self.av(x).permute(0,2,1)
        x = x.reshape(N,L,1)

        frequential = self.fdft(x, dim=1).squeeze(-1)
        frequential = torch.arcsinh(frequential)
        # frequential = torch.arcsinh(torch.mul(frequential, snr))

        # plt.scatter(t[0].detach().cpu().numpy(), sig[0].detach().cpu().numpy())
        # frequential = torch.arcsinh(frequential)
        # plt.plot(frequential[0].detach().cpu().numpy())
        # plt.show()
        # raise Exception()

        # frequential = torch.arcsinh(frequential * std)
        
        layone = self.layerone(frequential)
        # layone = self.norm(layone)
        layone = self.av(layone)
        layone = self.dp(layone)
        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        inter = self.dp(laytwo)
        inter = self.norm(inter)
        
        final = self.layerthree(inter)
        return final
    


class NUFourierModel(nn.Module):
    def __init__(self, samples, out, learnsamples=False):
        super().__init__()
        self.samples = samples
        self.out = out

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//4)
    
        
        self.layerthree = nn.Linear(samples//4,self.out)
        self.dp = nn.Dropout(p=0.8)
        
        
        self.fdft = FDFT(samples, learning=learnsamples, real=True)
        
        self.norm = nn.LayerNorm(samples//4)
        self.av = nn.ReLU()

    def forward(self, x):
        N = x.shape[0]
        L = x.shape[1]
        sig = x[:,:,0].reshape(N,L,1)
        std = torch.max(x[:,:,1], dim=-1)[0].reshape(N,1)
        t = x[:,:,2].reshape(N,L)
        frequential = self.fdft(sig, t, dim=1).squeeze(-1)
        # plt.scatter(t[0].detach().cpu().numpy(), sig[0].detach().cpu().numpy())
        # frequential = torch.arcsinh(frequential)
        # plt.plot(frequential[0].detach().cpu().numpy())
        # plt.show()
        # raise Exception()

        # frequential = torch.arcsinh(frequential * std)
        
        layone = self.layerone(frequential)
        # layone = self.norm(layone)
        layone = self.av(layone)
        layone = self.dp(layone)
        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        inter = self.dp(laytwo)
        inter = self.norm(inter)
        
        final = self.layerthree(inter)
        return final


class MultiFourierModel(nn.Module):
    def __init__(self, samples, out, learnsamples=False):
        super().__init__()
        self.samples = samples
        self.out = out

        self.ff1 = nn.Linear(samples,samples//2)
        self.ff2 = nn.Linear(samples,samples//2)

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//4)
    
        
        self.layerthree = nn.Linear(samples//4,self.out)
        self.dp = nn.Dropout(p=0.87)
        
        
        self.fdft = FDFT(samples, learning=learnsamples, real=False)
        
        self.norm = nn.LayerNorm(samples//4)
        self.av = nn.ReLU()

    def forward(self, x):
        N = x.shape[0]
        L = x.shape[1]
        sig = x[:,:,0].reshape(N,L,1)
        std = torch.max(x[:,:,1], dim=-1)[0].reshape(N,1)
        t = x[:,:,2].reshape(N,L)
        fft1 = self.fdft(sig, t, dim=1).squeeze()
        fft2 = torch.fft.fft(fft1)
        fft1 = torch.arcsinh(torch.abs(fft1) * std)
        fft2 = torch.arcsinh(torch.abs(fft2) * std)
        ff1 = self.ff1(fft1)
        ff2 = self.ff2(fft2)
        ff = torch.cat([ff1, ff2], dim=-1)
        frequential = self.dp(self.av(ff))
        

        layone = self.layerone(frequential)


        # layone = self.norm(layone)
        layone = self.av(layone)
        layone = self.dp(layone)
        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        inter = self.dp(laytwo)
        inter = self.norm(inter)
        
        
        final = self.layerthree(inter)
        return final

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
        
        
    def forward(self, x: torch.Tensor, t, dim=-1):
        fourier_tens = self.make_fourier(x.size(dim), t)
        temp = x.to(torch.cfloat)
        if len(temp.shape) == 2:
            transformed = torch.bmm(fourier_tens, temp.unsqueeze(-1)).squeeze(-1)
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
        fourier_tens = self.make_fourier(x.size(dim))
        temp = x.to(torch.cfloat)
        transformed = torch.matmul(fourier_tens, temp)
        if self.zerocomp:
            transformed[:,0] = 0
            transformed[:,-1] = 0
        return self.out(transformed)