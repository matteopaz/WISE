import torch
import numpy
import plotly.graph_objects as go
from genset import GenSet
import os
import shutil


many = 20
inst = GenSet()

def plot(ex, savestr):
    x = ex.detach().cpu().numpy()
    fig = go.Figure()
    tr = go.Scatter(x=x[:,2], y=x[:,0], mode="markers", marker=dict(color="black", opacity=0.5, size=4), name="Data")

    fig.add_trace(tr)
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            showgrid=True,
            showticklabels=True  # Set to True to show tick labels
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1,
            showgrid=True,
            showticklabels=True  # Set to True to show tick labels
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=800,
        height=600,
        autosize=False,
        plot_bgcolor='white'
    )
    fig.update_xaxes(range=[x[:,2].min(), x[:,2].max()])
    fig.update_yaxes(range=[x[:,0].min(), x[:,0].max()])

    fig.show()
    
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

