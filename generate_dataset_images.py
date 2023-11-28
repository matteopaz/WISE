import os
import sys
import numpy as np
import plotly.graph_objects as go
import pickle

ROOT = os.path.join("./")
sys.path.append(ROOT + "lib")

from plotly_style import update_layout

def get_trace(lightsource, key):

  x_data = lightsource["norm"]["day"]
  y_data = lightsource["norm"][key]

  sort_idxs = np.argsort(x_data)

  x_data = x_data[sort_idxs]
  y_data = y_data[sort_idxs]


  return go.Scatter(
    x=x_data,
    y=y_data,
    mode='markers',
    marker=dict(size=5, opacity=.75),
    name=key,
  )

def plot_from_lightsource(lightsource, ys=["w1"]):

  fig = go.Figure()

  for y in ys:
    fig.add_trace(get_trace(lightsource, y))

  update_layout(fig, legend_out=True)

  fig.layout.width = 800
  fig.layout.height = 0.65 * fig.layout.width

  return fig

def plot_from_tensor(data):
  fig = go.Figure()

  w1 = data[:, 0].numpy()
  std = data[:, 1].numpy()


  day = data[:, -1].numpy()


  fig.add_trace(go.Scatter(x=day, y=w1, marker=dict(size=5, opacity=0.7), name="w1mpro z-scored", mode='markers'))
  fig.add_trace(go.Scatter(x=day, y=std, marker=dict(size=5, opacity=0.7), name="w2mpro z-scored", mode='markers'))

  update_layout(fig, legend_out=True)

  fig.layout.width = 800
  fig.layout.height = 0.65 * fig.layout.width


  return fig

# ----------------------- Begin Work on Dataset ----------------------- # 

buckets = pickle.load(open(ROOT + "cached_data/raw_classes.pkl", "rb"))


for kind in buckets:
    for objname in buckets[kind]:
        lightsource = buckets[kind][objname]
        
        fig = plot_from_lightsource(lightsource, ["w1", "w2"])
        
        fig.update_layout(
            title="{} example - {}".format(kind, objname),
            xaxis_title="Normalized days",
            yaxis_title="Band magnitudes",
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="Black"
            )
        )
        
        fig.write_image(ROOT + "dataset_imgs/{}/{}.jpg".format(kind, objname))

        
