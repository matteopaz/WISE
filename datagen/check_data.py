import torch
import numpy
import plotly.graph_objects as go
from genset import GenSet
import os
import shutil


many = 10
inst = GenSet()

def plot(ex, savestr):
    x = ex.detach().cpu().numpy()
    fig = go.Figure()
    tr = go.Scatter(x=x[:,2], y=x[:,0], mode="markers", marker=dict(color="black", opacity=0.5, size=5), name="Data")
    # trerr = go.Scatter(x=x[:,2], y=x[:,1], mode="markers", marker=dict(color="black", opacity=0.5, size=5), name="Uncertainty")
    fig.add_trace(tr)
    # fig.add_trace(trerr)
    fig.write_image("./imgs/{}.png".format(savestr))

# clear the directory
shutil.rmtree("./imgs")
os.mkdir("./imgs")

# null
for i in range(many):
    i = str(i)
    ex = inst.gen_null()
    plot(ex, "null" + i)

for i in range(many):
    i = str(i)
    ex, pd = inst.gen_transit(returnpd=True)
    plot(ex, "transit" + i)
    ex[:,2] = ex[:,2] % pd
    plot(ex, "transitfolded" + i)

for i in range(many):
    i = str(i)
    ex, pd, ph = inst.gen_pulsating_var(returnpd=True)
    plot(ex, "pulsating" + i)
    ex[:,2] = (ex[:,2] - ph) % pd
    plot(ex, "pulsatingfolded" + i)
    

for i in range(many):
    i = str(i)
    ex = inst.gen_nova()
    plot(ex, "nova" + i)

