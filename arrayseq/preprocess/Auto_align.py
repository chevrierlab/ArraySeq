
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters, feature, color, io
from scipy.spatial import KDTree
from tqdm import tqdm
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu, gaussian
import skimage
import pkg_resources

def remove_small_object(image, sigma=1, manual_threshold_intensity=None, object_connectivity=2, min_tissue_size=0.001):
    gray = rgb2gray(image)
    blurred = gaussian(gray, sigma=sigma)
    if manual_threshold_intensity is None:
        threshold = skimage.filters.threshold_otsu(blurred) * 1.1
    elif manual_threshold_intensity != 'auto':
        threshold = float(manual_threshold_intensity)
    binary_mask = blurred < threshold
    min_size = image.size * min_tissue_size
    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size, connectivity=object_connectivity)
    return cleaned_mask
