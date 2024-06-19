
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from skimage import exposure
from matplotlib.cm import ScalarMappable

def plot_2d(data, colormap='viridis'):
    cmap = LinearSegmentedColormap.from_list("custom_cmap", plt.cm.get_cmap(colormap).colors, N=len(data))
    norm = Normalize(vmin=min(data), vmax=max(data))
    sm = ScalarMappable(cmap=cmap, norm=norm)
    plt.scatter(range(len(data)), data, c=sm.to_rgba(data))
    plt.colorbar(sm)
    plt.show()
