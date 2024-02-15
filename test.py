import torch
import os
import sys
import plotly.graph_objects as go
from math import sqrt
import numpy as np

shortspacing = 1 # days
longspacing = 29 # days between obs groups

def apparitionsbeforegap():
    options = [10, 10, 10, 10, 10, 10, 14, 14, 14, 14, 20, 20, 20, 30, 30, 50, 100, 200, 100000]
    return np.random.choice(np.array(options))
    
days = 250

buckets = {
    "null": [],
    "nova": [],
    "pulsating_var": [],
    "transit": []
}

def gen_sampling():
    x = [0]
    i = 0
    app = apparitionsbeforegap()
    while x[-1] < days:
        if i % app == 0:
            x.append(x[-1] + longspacing)
        else:
            x.append(x[-1] + shortspacing)
        i += 1
    return torch.tensor(x)

torch.set_default_device("cpu") 

sys.path.append("./models")
from fourier import FUDFT, FDFT

l = days
# inst = FUDFT(l, real=True, remove_zerocomp=False)
  
# t = torch.linspace(0,1,l+100)[torch.randperm(l+100)][100:] 
# t = torch.sort(t).values
t = gen_sampling()
t = t

x = torch.sin(2 * torch.pi * t / l)
 
tch = (1 / sqrt(l)) * torch.abs(torch.fft.fft(x)).to(torch.float32) 
col = -2 * torch.pi * 1j * t * (l - 1) / (l * torch.max(t))
row = torch.arange(l)
exp = torch.outer(row, col) 
four = torch.exp(exp)
mat = (1 / sqrt(l))*torch.matmul(four, x.to(torch.cfloat))
mat = torch.abs(mat)





fig = go.Figure()
tr1 = go.Scatter(
    x=t,
    y=x,
    mode="markers",
    name="signal"
)
tr2 = go.Scatter(
    x=t,
    y=mat,
    mode="lines",
    name="mat"
)
tr3 = go.Scatter(
    x=t,
    y=tch,
    mode="lines",
    name="tch"
)
tr4 = go.Scatter(
    x=np.linspace(0, torch.max(t), len(t)), 
    y=x,
    mode="markers",
    name="Equispaced signal"
)

fig.add_trace(tr1)
fig.add_trace(tr2)
fig.add_trace(tr3)
fig.add_trace(tr4)
fig.write_html("temp.html")