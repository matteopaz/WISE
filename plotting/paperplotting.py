import plotly.graph_objects as go
import sys
import numpy as np

sys.path.append('./lib')
from plotly_style import update_layout
# make a graph of 4 dirac impulses on a grid of 100 points
fig = go.Figure()

grid = np.zeros(100)
# add 3 random dirac impulses
imps = [(20,3), (40, -3), (60, 2), (80, -2)]
for imp in imps:
    grid[imp[0]] = imp[1]

# dotted line
d = lambda x,h: go.Scatter(x=[x,x], y=[0,h], mode="lines", line=dict(color="blue", width=2, dash="dash"))
# add open circle at top of line
c = lambda x,h: go.Scatter(x=[x], y=[h], mode="markers", marker=dict(color="white", size=4, symbol="circle", line=dict(color="blue", width=1)))

# convolve the grid with a gaussian
def convolve(grid, sigma):
    x = np.linspace(0, 100, 100)
    y = np.zeros(100)
    for i in range(100):
        y[i] = np.sum(grid*np.exp(-(x-x[i])**2/(2*sigma**2)))
    return y

fig.add_trace(go.Scatter(x=np.linspace(0, 100, 100), y=convolve(grid, 10), mode="lines", line=dict(color="red", width=2)))


# for imp in imps:
#     fig.add_trace(d(imp[0], imp[1]))
#     fig.add_trace(c(imp[0], imp[1]))

# # add gaussians around the dirac impulses

# def g(imp):
#     x = np.linspace(0, 100, 1000)
#     return imp[1]*np.exp(-(x-imp[0])**2/100)

# for imp in imps:
#     fig.add_trace(go.Scatter(x=np.linspace(0, 100, 1000), y=g(imp), mode="lines", line=dict(color="orange", width=2, dash="dot")))


update_layout(fig)
# no legend
fig.update_layout(title="", showlegend=False)
# x-axis from 0 to 100
fig.update_xaxes(range=[0, 100])
# y-axis from 0 to 100
fig.update_yaxes(range=[-3.25, 3.25])
# hide axis values and grid
fig.update_xaxes(showticklabels=False, showgrid=False)
fig.update_yaxes(showticklabels=False, showgrid=False)
# plot x axis with arrow
fig.add_shape(type="line", x0=0, x1=100, y0=0, y1=0, line=dict(color="black", width=2))

fig.write_image("./4dirac.png")