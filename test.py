import pywt
import numpy as np
import matplotlib.pyplot as plt

brightness = 1
std = 0.1

g = lambda x, std: np.random.normal(x, std)

s = lambda x: np.random.uniform(x[0], x[1])

selector = np.random.random()
if selector > 0.7: # 30% SP
    period = s([0.1, 2])
elif selector > 0.2: # 50% MP
    period = s([2, 100])
else: # 20% LP
    period = s([100, 400])

selector = np.random.random()
max_amp = s([3*std/brightness, 0.75])

gridres = 110
grid = np.zeros(gridres)

windowsize = gridres // 2.7

modifier = windowsize * 6
filt = (1 / np.sqrt(np.pi*modifier)) * np.exp(-(1 / modifier)*np.arange(-windowsize, windowsize)**2)
# filt[:int(windowsize)] += 0.001
# if selector > 0.4: # 60% mountainrange waveform
n_peaks = int(np.abs(g(0,4)) + 2)

centers = []
for i in range(n_peaks):
    centers.append(int(s([gridres*i / n_peaks + windowsize/2, gridres*(i+1) / n_peaks - windowsize/2])))
heights = [(-1)**k * s([0, max_amp]) for k in range(n_peaks)] 
grid[centers] = heights

padding = np.zeros(int(windowsize//3))
grid = np.concatenate((padding, grid, padding))

grid = np.convolve(grid, filt, mode="same")
# relative to eachother peaks. Biggest peak can be 4.5 times bigger than smallest
# Also alternates signs with (-1)**k

grid = grid * max_amp / np.max(np.abs(grid)) # rescale to max_amp

plt.plot(grid)
plt.savefig("test.png")