import gzip
import tdqm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from skimage.color import rgb2gray
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu, gaussian
import pkg_resources

def remove_small_object(image, sigma=1, manual_threshold_intensity=None, object_connectivity=2, min_tissue_size=0.001):
    gray = rgb2gray(image)
    blurred = gaussian(gray, sigma=sigma)
    if manual_threshold_intensity is None:
        threshold = threshold_otsu(blurred) * 1.1
    elif manual_threshold_intensity != 'auto':
        threshold = float(manual_threshold_intensity)
    binary_mask = blurred < threshold
    min_size = image.size * min_tissue_size
    object_mask = remove_small_objects(binary_mask, min_size, connectivity=object_connectivity)
    return object_mask


def map_barcodes(adata, Barcode_coord_file=None):

    adata_x = adata.copy()

    if Barcode_coord_file is None:
        Barcode_coord_file = pkg_resources.resource_filename('arrayseq', './_data/Barcodes/ArraySeq_n12_mapping.csv.gz')

    with gzip.open(Barcode_coord_file, 'rt') as f:
        coords = pd.read_csv(f)
    
    adata_x.obs["Barcode"] = adata_x.obs.index
    adata_x.obs = adata_x.obs.join(coords.set_index("Barcode"), on="Barcode")

    adata_x.obs["X"], adata_x.obs["Y"] = adata_x.obs["X"]*100, adata_x.obs["Y"]*100

    return adata_x


# Function to apply transformation (translation and scaling)
def transform_points(points, tx, ty, scale):
    return points * scale + np.array([tx, ty])

# Function to bin the binary mask
def bin_mask(binary_mask, bin_size=15, threshold=0.75):
    mask_points = []
    rows, cols = binary_mask.shape
    for i in range(0, rows, bin_size):
        for j in range(0, cols, bin_size):
            bin_region = binary_mask[i:i+bin_size, j:j+bin_size]
            if np.mean(bin_region) >= threshold:
                mask_points.append([j + bin_size//2, i + bin_size//2])
    return np.array(mask_points)

# Function to downsample x and y data points
def downsample_data(adata, n_HE_points):
    sampled_df = adata.obs.sample(n_HE_points, replace=False, random_state=43)
    return sampled_df

def downsample_HE(data, n_points_removed):
    sample_indices = np.random.choice(data.shape[0], size=data.shape[0] - n_points_removed, replace=False)

    # Sample the rows using the indices
    downsampled_data = data[sample_indices]

    return downsampled_data

# Residuals function to compute the difference between transformed points and the closest mask points
def residuals_with_progress(params, points, mask_points, progress_bar):
    tx, ty, scale = params
    transformed_points = transform_points(points, tx, ty, scale)
    distances = cdist(transformed_points, mask_points)
    progress_bar.update(1)
    return np.min(distances, axis=0)

# Function to detect and remove outliers
def remove_outliers(data_points, mask_points, threshold=0):
    initial_params = [0, 0, 1]
    n_iterations = 500  # Number of iterations for the progress bar

    with tqdm(total=n_iterations) as pbar:
        result = least_squares(
            residuals_with_progress,
            initial_params,
            args=(data_points, mask_points, pbar),
            max_nfev=n_iterations
        )
    optimized_tx, optimized_ty, optimized_scale = result.x
    transformed_points = transform_points(data_points, optimized_tx, optimized_ty, optimized_scale)
    distances = cdist(transformed_points, mask_points)
    min_distances = np.min(distances, axis=1)
    squared_distances = min_distances**2
    mean_distance = np.mean(squared_distances)
    std_distance = np.std(squared_distances)
    outliers = squared_distances > (mean_distance + threshold * std_distance)
    filtered_data_points = data_points[~outliers]
    return filtered_data_points, sum(outliers)

def residuals_final_with_progress(params, points, mask_points, progress_bar):
    tx, ty, scale = params
    transformed_points = transform_points(points, tx, ty, scale)
    distances = cdist(transformed_points, mask_points)
    progress_bar.update(1)
    return np.min(distances, axis=0)

def final_alignment_with_progress(filtered_downsample_spatial, detected_HE, initial_params, bounds):
    n_iterations = 500  # Number of iterations for the progress bar

    with tqdm(total=n_iterations) as pbar:
        result = least_squares(
            residuals_final_with_progress,
            initial_params,
            args=(filtered_downsample_spatial, detected_HE, pbar),
            bounds=bounds,
            max_nfev=n_iterations
        )

    optimized_tx, optimized_ty, optimized_scale = result.x
    return optimized_tx, optimized_ty, optimized_scale

def aulto_align(
    adata,
    HE_image,
    UMI_threshold=1000,
    sigma=2,
    object_connectivity=2,
    min_tissue_size=0.01,
    bin_size=30,
    threshold=0.75,
    downsample_random_state=42,
    outlier_threshold=0,
    initial_tx=0,
    initial_ty=0,
    initial_scale=1,
    bounds=([-5000, -10000, 1], [750, 1000, 6]), 
    show_image = True, 
    image_size = 6, 
    point_color = "#3897f5", 
    point_alpha = 0.8, 
    point_size = 2
):
    
    """
    Aligns spatial transcriptomics data stored in an AnnData object to a histology image using image processing and optimization techniques.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing spatial transcriptomics data with coordinates in 'obs' DataFrame.
    HE_image : np.ndarray
        A 2D histology image (H&E stained) to which the spatial data will be aligned.
    UMI_threshold : int, optional
        Minimum number of UMIs for a spot to be included in the alignment process, by default 1000.
    sigma : float, optional
        Standard deviation for Gaussian kernel used in image preprocessing, by default 2.
    object_connectivity : int, optional
        Connectivity for identifying connected components in the binary mask, by default 2.
    min_tissue_size : float, optional
        Minimum size of tissue to be retained after removing small objects, specified as a fraction of the image size, by default 0.01.
    bin_size : int, optional
        Size of the bins used for binning the binary mask, by default 30.
    threshold : float, optional
        Threshold for binarizing the image, by default 0.75.
    downsample_random_state : int, optional
        Random state for reproducibility when downsampling the spatial data, by default 42.
    outlier_threshold : float, optional
        Threshold for removing outliers from the downsampled spatial data, by default 0.
    initial_tx : float, optional
        Initial translation in the x-direction for optimization, by default 0.
    initial_ty : float, optional
        Initial translation in the y-direction for optimization, by default 0.
    initial_scale : float, optional
        Initial scale factor for optimization, by default 1.
    bounds : tuple of lists, optional
        Bounds for the optimization parameters in the format (min_bounds, max_bounds), by default ([-5000, -10000, 1], [750, 1000, 6]).

    Returns
    -------
    AnnData
        A new AnnData object with the spatial coordinates in 'obs' updated to align with the histology image.

    Notes
    -----
    The function performs the following steps:
    1. Preprocesses the input histology image to create a binary mask.
    2. Bins the binary mask to detect tissue regions.
    3. Filters and downsamples the spatial transcriptomics data.
    4. Removes outliers from the downsampled spatial data.
    5. Optimizes translation and scaling parameters to align the spatial data with the histology image.
    6. Applies the optimized transformation to all spatial coordinates in the AnnData object.

    Example
    -------
    >>> aligned_adata = aulto_align(adata=adata, HE_image=he_image, UMI_threshold=500)
    """
    # Copy the adata object to avoid modifying the original
    adata_copy = adata.copy()
    adata_copy = map_barcodes(adata_copy)
    # Preprocess the image
    binary_mask = remove_small_object(HE_image, sigma=sigma, object_connectivity=object_connectivity, min_tissue_size=min_tissue_size)
    plt.imshow(binary_mask)
    plt.show()
    detected_HE = bin_mask(binary_mask, bin_size=bin_size, threshold=threshold)

    # Simulate adata processing
    adata_copy.obs["UMI"] = np.concatenate(adata_copy.X.sum(axis=1).tolist()).flat
    adata_filtered = adata_copy[adata_copy.obs["UMI"] > UMI_threshold]
    downsample_spatial = downsample_data(adata_filtered, detected_HE.shape[0])
    downsample_spatial = downsample_spatial[["X", "Y"]].to_numpy()

    # Remove outliers from the data points
    filtered_downsample_spatial, n_points_removed = remove_outliers(downsample_spatial, detected_HE, outlier_threshold)

    print(f"Number of points removed as outliers: {n_points_removed}")

    # Adjust detected_HE for removed outliers
    detected_HE = downsample_HE(detected_HE, n_points_removed)

    # Re-run optimization without outliers
    initial_params = [initial_tx, initial_ty, initial_scale]
    optimized_tx, optimized_ty, optimized_scale = final_alignment_with_progress(filtered_downsample_spatial, detected_HE, initial_params, bounds)

    # Get the optimized parameters    
    print("Optimized translation (tx, ty):", optimized_tx, optimized_ty)
    print("Optimized scale:", optimized_scale)

    # Transform data points using the optimized parameters
    aligned_points = transform_points(downsample_spatial, optimized_tx, optimized_ty, optimized_scale)

    # Apply the transformation to all X and Y data points in adata
    all_points = adata_copy.obs[["X", "Y"]].to_numpy()
    transformed_all_points = transform_points(all_points, optimized_tx, optimized_ty, optimized_scale)
    adata_copy.obs["X"] = transformed_all_points[:, 0]
    adata_copy.obs["Y"] = transformed_all_points[:, 1]

    if show_image == True:
        fig, ax = plt.subplots(figsize=(image_size * 1.5, image_size))
        ax.imshow(rgb2gray(HE_image), cmap='gray')
        scatter = ax.scatter(adata_copy.obs["X"].tolist(), 
                             adata_copy.obs["Y"].tolist(), 
                             c=point_color, 
                             s=point_size, 
                             alpha=point_alpha)
        
        ax.set_title("Aligned ST")
        plt.show()


    return adata_copy
