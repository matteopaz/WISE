"""Project: Plotly Scientific Plot Styling
Version: 1.0
Author: miladiouss
Date: Aug 28, 2023
Description: Enhance appearance of plotly figures to meet publication-quality standards.
Usage:
  with open("plotly-style.yaml", "r") as stream:
      style_dict = yaml.safe_load(stream)
  fig.update(style_dict)
"""
import plotly.graph_objects as go


def update_layout(
    fig: go.Figure, grid=True, constant_legend_itemsizing=False, legend_out=False
):
    """Enhance appearance of plotly figures by updating its layout.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        A plotly figure object
    """

    # Set Font Family
    fig.layout.font.family = "Times New Roman"
    fig.layout.legend.font.family = "monospace"

    # Set Font Size
    fig.layout.font.size = 18
    fig.layout.title.font.size = fig.layout.font.size
    fig.layout.xaxis.title.font.size = fig.layout.font.size
    fig.layout.yaxis.title.font.size = fig.layout.font.size
    fig.layout.legend.font.size = fig.layout.font.size - 3

    # Configure Plot Size
    fig.layout.width = 750
    fig.layout.height = 500
    # Configure Colors
    fig.layout.plot_bgcolor = "rgba(0, 0, 0, 0)"

    # Configure Legend
    fig.layout.showlegend = True
    fig.layout.legend.bgcolor = "rgba(0, 0, 0, 0)"
    if constant_legend_itemsizing:
        fig.layout.legend.itemsizing = "constant"

    # Legend Position
    if legend_out:
        fig.layout.legend.xanchor = "left"
        fig.layout.legend.x = 1.02

    else:
        fig.layout.legend.xanchor = "right"
        fig.layout.legend.x = 1

    # Set Margins
    m = 75
    fig.layout.margin.t = m
    fig.layout.margin.b = m
    fig.layout.margin.l = m + 25
    fig.layout.margin.r = m + 25
    fig.layout.margin.pad = 0

    # Configure Axis Lines
    fig.layout.xaxis.linecolor = "black"
    fig.layout.yaxis.linecolor = "black"
    fig.layout.xaxis.mirror = True
    fig.layout.xaxis.showline = True
    fig.layout.yaxis.mirror = True
    fig.layout.yaxis.showline = True
    fig.layout.xaxis.zeroline = False
    fig.layout.yaxis.zeroline = False

    # Configure Ticks
    fig.layout.xaxis.ticks = "inside"
    fig.layout.yaxis.ticks = "inside"
    # Configure Minor Ticks
    if grid:
        fig.layout.xaxis.minor.ticks = "inside"
        fig.layout.yaxis.minor.ticks = "inside"
        fig.layout.xaxis.minor.ticklen = 1
        fig.layout.yaxis.minor.ticklen = 1

    # Configure Grid
    fig.layout.xaxis.gridcolor = "rgba(0, 0, 0, .15)"
    fig.layout.yaxis.gridcolor = "rgba(0, 0, 0, .15)"
    fig.layout.xaxis.zerolinecolor = "rgba(0, 0, 0, .15)"
    fig.layout.yaxis.zerolinecolor = "rgba(0, 0, 0, .15)"
    fig.layout.xaxis.minor.showgrid = grid
    fig.layout.yaxis.minor.showgrid = grid
    fig.layout.xaxis.showgrid = grid
    fig.layout.yaxis.showgrid = grid
    fig.layout.xaxis.minor.gridcolor = "rgba(1, 1, 1, .05)"
    fig.layout.yaxis.minor.gridcolor = "rgba(1, 1, 1, .05)"


def add_caption(fig, text="Figure 1. Example Data Visualization."):
    n_lines = len(text.split("<br>"))
    fig.layout.margin.b += 20 * n_lines
    fig.layout.height += 20 * n_lines
    _ = fig.add_annotation(
        dict(
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top",
            align="left",
            x=0.5,
            y=-0.175,
            showarrow=False,
            text=text,
            font=dict(size=16),
        )
    )


def web_friendly_display(fig):
    from IPython.display import SVG

    return SVG(fig.to_image(format="svg").decode("utf-8").replace("−", "-"))
