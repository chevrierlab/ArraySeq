
import plotly.graph_objs as go
import matplotlib.cm as cm
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
import plotly.io as pio
import random

def generate_custom_cmap(num_colors):
    # Generate a custom colormap with `num_colors` distinct colors.
    colors = cm.nipy_spectral(np.linspace(0, 1, num_colors))
    return [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c in colors]

def convert_matplotlib_to_plotly_cmap(cmap_name, num_colors):
    # Convert a matplotlib colormap to a plotly colorscale
    cmap = cm.get_cmap(cmap_name, num_colors)
    colors = [to_rgb(cmap(i)) for i in range(cmap.N)]
    plotly_colorscale = [[i/(len(colors)-1), f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'] for i, c in enumerate(colors)]
    return plotly_colorscale

def generate_random_cmap(num_colors=8):
    # Generate a user-defined colormap with `num_colors` distinct random colors.
    # Returns a dictionary where keys are category labels and values are RGB strings.
    cmap = {}
    for i in range(num_colors):
        r, g, b = [random.randint(0, 255) for _ in range(3)]
        cmap[f"Category {i+1}"] = f"rgb({r}, {g}, {b})"
    return cmap
