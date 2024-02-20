import os
import sys
import numpy as np
import plotly.graph_objects as go
import pickle

ROOT = os.path.join("./")
sys.path.append(ROOT + "lib")

from plotly_style import update_layout

def get_trace(lightsource, key, norm=True):
  basekey = "norm" if norm else "raw"
  x_data = lightsource[basekey]["day"]
  y_data = lightsource[basekey][key]

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

def plot_from_lightsource(lightsource, ys=["w1"], norm=True):

  fig = go.Figure()

  for y in ys:
    fig.add_trace(get_trace(lightsource, y, norm))

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

  update_layout(fig, legend_out=True)

  fig.layout.width = 800
  fig.layout.height = 0.65 * fig.layout.width


  return fig

# ----------------------- Begin Work on Dataset ----------------------- # 

buckets = pickle.load(open(ROOT + "cached_data/raw_classes.pkl", "rb"))



for kind in buckets:
    # clean out all items in folder
    os.system(f"rm -rf {ROOT}/dataset_imgs/{kind}")
    os.system(f"mkdir -p {ROOT}/dataset_imgs/{kind}")
    for objname in buckets[kind]:
        lightsource = buckets[kind][objname]
        
        fig = go.Figure()
        tr = go.Scatter(
          x = lightsource["raw"]["mjd"],
          y = lightsource["raw"]["w1"],
          mode='markers',
          marker=dict(size=5, opacity=.75, color="black"),
          name="w1flux"

        )
        fig.add_trace(tr)
        
        fig.update_layout(
            title="{} example - {}".format(kind, objname),
            xaxis_title="Normalized days",
            yaxis_title="W1 Flux",
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="Black"
            )
        )



        print(f'Saving image to: {os.path.abspath(ROOT + "dataset_imgs/{}/{}.jpg".format(kind, objname))}')
        fig.write_image(ROOT + "dataset_imgs/{}/{}.jpg".format(kind, objname))

        
