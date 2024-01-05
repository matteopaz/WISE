import numpy as np
import torch
import torch.nn as nn

class FourierModelTest(nn.Module):
    def __init__(self, samples, out):
        super().__init__()
        self.samples = samples
        self.out = out

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//8)
        self.layerthree = nn.Linear(samples//8,self.out)
        self.dp = nn.Dropout(p=0.2)
        
        self.av = nn.Sigmoid()
        self.prob = nn.LogSoftmax(dim=1)
        

    def make_fourier(self, N):
        freqs = np.linspace(start=0, stop=N-1, num=self.samples)
        access_freq = np.vectorize(lambda k: freqs[int(k)])
        unity = lambda k: np.exp(-2*np.pi*1j*k / N)
        fourier = np.fromfunction(lambda i,j: unity(access_freq(i)*j), (self.samples, N))
        return torch.from_numpy(fourier)

    def forward(self, x):
        x = x[:, :, 0].unsqueeze(2)
        x = x.to(dtype=torch.cfloat)
        N = x.shape[0]
        T = x.shape[1]
        fourier = self.make_fourier(T).repeat(N,1,1).to(dtype=torch.cfloat)
        frequential = torch.bmm(fourier, x).squeeze()
        frequential = frequential.abs().to(dtype=torch.float32) # compress complex to float by mag
        frequential[:,0] = torch.zeros_like(frequential[:,0])
        layone = self.layerone(frequential)
        layone = self.av(layone)
        layone = self.dp(layone)
        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        laytwo = self.dp(laytwo)
        final = self.layerthree(laytwo)
        return final

class FourierModelTestNU(nn.Module):
    def __init__(self, samples, out):
        super().__init__()
        self.samples = samples
        self.out = out

        self.layerone = nn.Linear(samples,samples//2)
        self.layertwo = nn.Linear(samples//2,samples//8)
        self.layerthree = nn.Linear(samples//8,self.out)
        self.dp = nn.Dropout(p=0.4)
        
        self.av = nn.Sigmoid()
        self.prob = nn.LogSoftmax(dim=1)
        

    def make_fourier(self, time):
        # want Batch x Samples x Time
        N = time.shape[0]
        T = time.shape[1]
        freqs = np.linspace(start=0, stop=T-1, num=self.samples)
        frequency_panes = []
        for f in freqs:
            pane = np.exp(-2*np.pi*1j*f*time)
            frequency_panes.append(pane)
        fourier = np.stack(frequency_panes, axis=1)
        return torch.from_numpy(fourier)
            

    def forward(self, inp):
        x = inp[:, :, 0].unsqueeze(2)
        day = inp[:, :, 2]
        std = inp[0,0,1]
        x = x.to(dtype=torch.cfloat)
        N = inp.shape[0]
        T = inp.shape[1]
        fourier = self.make_fourier(day)
        frequential = torch.bmm(fourier, x).squeeze()
        frequential = frequential.abs().to(dtype=torch.float32) # compress complex to float by mag
        layone = self.layerone(frequential)
        layone = self.av(layone)
        layone = self.dp(layone)
        laytwo = self.layertwo(layone)
        laytwo = self.av(laytwo)
        laytwo = self.dp(laytwo)
        final = self.layerthree(laytwo)
        return final
    

class FFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft = torch.fft.FFT()
        
    def forward(self, x):
        mag = x[:, :, 0]
        fft = self.fft(mag, dim=2)
    
        
        