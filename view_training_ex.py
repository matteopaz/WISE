import os
import sys
import numpy as np
import plotly.graph_objects as go
import pickle
import torch

ROOT = os.path.join("./")
sys.path.append(ROOT + "lib")

from plotly_style import update_layout
from helpers import plot_from_tensor
from sourceset import SourceSet


# ----------------------- Begin Work on Dataset ----------------------- # 

with open(ROOT + "processed_datasets/pseudotrain.pt", "rb") as f:
  train = torch.load(f)


for _, (exbatch, l) in enumerate(train):
    # if (torch.abs(ex) > 5).any():
        
    for i, ex in enumerate(exbatch):
        
        fig = go.Figure()
        y = ex[:, 0].numpy()
        unc = ex[:, 1].numpy()
        x = ex[:, 2].numpy()
        tr = go.Scatter(x=x, y=y, mode="markers", error_y=dict(type="data", array=unc, visible=True))
        fig.add_trace(tr)
        class_ = torch.argmax(l[i]).item()
            
        fig.update_layout(
            title=f"{i}th example - {class_}",
            xaxis_title="Normalized days",
            yaxis_title="Band magnitudes",
            font=dict(
                family="Courier New, monospace",
                size=14,
                color="Black"
            )
        )

        i += 1

        print(f'Saving image to: {os.path.abspath(ROOT + f"dataset_imgs/rawtrain/{i}-{class_}.jpg")}')
        fig.write_image(os.path.abspath(ROOT + f"dataset_imgs/rawtrain/{i}-{class_}.jpg"))

