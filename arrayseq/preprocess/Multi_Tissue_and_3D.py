
import warnings
import pandas as pd
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label
from skimage.morphology import remove_small_objects, binary_closing, disk, binary_erosion, binary_dilation, closing
from skimage.util import invert
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from tqdm import tqdm
import skimage
import os
import re
import gc
import math
import scanpy as sc
import glob

warnings.filterwarnings('ignore')

def Remove_Background_Tissue(image, tissue_min_size, tissue_sigma, tissue_threshold_rescale_factor, tissue_connectivity):
    # Convert image to grayscale
    gray = rgb2gray(image)

    # Apply Gaussian blur
    blurred = gaussian(gray, sigma=tissue_sigma)
    threshold = threshold_otsu(blurred) * tissue_threshold_rescale_factor
    binary_mask = blurred < threshold

    # Remove small objects
    cleaned_mask = remove_small_objects(binary_mask, min_size=tissue_min_size, connectivity=tissue_connectivity)
    return cleaned_mask
