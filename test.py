import torch
import numpy as np
import plotly.graph_objects as go
import pickle
import sys
sys.path.append("./lib")
from lightsource import LightSource



with open("cached_data/train_buckets.pkl", "rb") as f:
   exs = pickle.load(f) 

ex = exs["nova"][0]
print(ex.name)

mjd = ex["norm"]["mjd"]
# time = np.arange(0, len(ex["raw"]["w1"]))
signal = ex["norm"]["w1"]

time = ex["norm"]["day"]
# N = len(time)

# signal = ex["norm"]["w1"]

# time = np.linspace(0, 2*np.pi, 1000)
N = len(time)
# signal = np.sin(time)

unity = np.exp(-2*np.pi*1j / N)

access = lambda i: time[int(i)] * N
# access = lambda i: i
access = np.vectorize(access)

dftm = np.fromfunction(lambda i,j: unity**(i*access(j)), (N,N))

dft = np.matmul(dftm, signal.reshape(N,1))


realdft = np.fft.fftshift(np.fft.fft(signal))[1:]


tr1 = go.Scatter(
    # x= (mjd) % 2.780714,
    x= (mjd),
    y=signal,
    mode="markers",
    # style markers
    marker=dict(
        size=3,
        color="blue",
    ),
)

tr2im = go.Scatter(
    x=time,
    y=np.imag(dft).reshape(N)[1:],
    mode="lines",
    line=dict(dash="dot"),
    name="Imaginary"
)

tr2rl = go.Scatter(
    x=time,
    y=np.real(dft).reshape(N)[1:],
    mode="lines",
    name="Real"
)

tr2mag = go.Scatter(
    # x=(np.arange(len(dft))/(mjd[-1]-mjd[0])), #period
    x=(np.arange(len(dft))), #period
    y=np.abs(dft).reshape(N)[1:],
    mode="lines",
    name="Magnitude"
)

samples = len(signal)
elapsed = mjd[-1]-mjd[0]
sampling_rate = samples/elapsed

biggest_bucket = np.argmax(dft)
freq = biggest_bucket * sampling_rate / len(dft)
period = 1/freq

print("samples: ", samples)
print("elapsed: ", elapsed)
print("sampling rate:", sampling_rate)
print("biggest bucket: ", biggest_bucket)
print("freq: ", freq)
print("period: ", period)


fig = go.Figure()
fig.add_trace(tr1)
fig.show()
fig2= go.Figure()
fig2.add_trace(tr2im)
fig2.add_trace(tr2rl)
fig2.add_trace(tr2mag)
fig2.show()

# fig3 = go.Figure()
# tr3 = go.Scatter(
#     x=list(range(len(realdft))),
#     y=np.abs(realdft).reshape(N),
#     mode="lines",
#     name="Magnitude"
# )
# fig3.add_trace(tr3)
# fig3.show()