import torch
import numpy as np
import pywt
import ptwt  # use "from src import ptwt" for a cloned the repo

# generate an input of even length.
data = np.arange(100)
data_torch = torch.from_numpy(data.astype(np.float32))
waveletname = input("wavelet: ")
wavelet = pywt.Wavelet(waveletname)

# compare the forward fwt coefficients
# print(pywt.wavedec(data, wavelet, mode='zero', level=1))
c, d = ptwt.wavedec(data_torch, wavelet, mode='zero', level=1)
print(len(data))
print(c,d)
print(c.shape, d.shape)

# invert the fwt.
# print(ptwt.waverec(ptwt.wavedec(data_torch, wavelet, mode='zero'), wavelet))