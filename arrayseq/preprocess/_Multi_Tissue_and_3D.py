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

def Remove_Background_Tissue(image, 
                             tissue_min_size, 
                             tissue_sigma, 
                             tissue_threshold_rescale_factor, 
                             tissue_connectivity):
    # Convert image to grayscale
    gray = rgb2gray(image)

    # Apply Gaussian blur
    blurred = gaussian(gray, sigma=tissue_sigma)

    # Threshold the image
    
    if threshold_otsu(blurred)*tissue_threshold_rescale_factor > 0.999:
        raise ValueError("Threshold intensity for tissue detection exceeds 1. Lower 'tissue_threshold_rescale_factor'.")
        
    else: 
        bw = blurred > threshold_otsu(blurred)*tissue_threshold_rescale_factor

    
    # Remove small objects
    bw = remove_small_objects(bw, min_size=tissue_min_size, connectivity = tissue_connectivity)

    # Invert the binary image
    bw = invert(bw)

    # Extract the largest object
    labels = label(bw)
    largest_object = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

    # Close the mask to merge all objects into a single object
    largest_object = binary_closing(largest_object, disk(10))

    # Create a copy of the image with all pixels set to white
    result = np.ones_like(image) * 255

    # Set the detected region to its original RGB values
    result[largest_object] = image[largest_object]

    # Set the background to white
    result[~largest_object] = [255, 255, 255]

    return result



def can_be_converted_to_int(value):
    return np.isclose(value, int(value))

def seperate_tissues(adata, 
                     seperate_column, 
                     adata_output_folder, 
                     sample_name, 
                     aspect = "auto", 
                     tissue_margin_buffer = 1.35, 
                     remove_background_tissue = True, 
                     tissue_min_size=5000, 
                     tissue_sigma=3, 
                     tissue_threshold_rescale_factor = 1.1, 
                     tissue_connectivity = 2,
                     preview_alignment = True, 
                     preview_images_pt_size = 1, 
                     preview_images_alpha = 0.5, 
                     preview_HE_images = True): 

    """
    Separate tissues in an AnnData object based on a specified column, crop the corresponding histological image, and optionally remove background tissue.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing spatial coordinates and histological image data.
    seperate_column : str
        Column name in `adata.obs` used to separate tissues.
    adata_output_folder : str
        Directory path to save the separated AnnData objects.
    sample_name : str
        Base name for the output files.
    aspect : str or float, optional, default="auto"
        Aspect ratio for the cropped image. If "auto", the aspect ratio is calculated from the data.
        If a float, it specifies the desired aspect ratio.
    tissue_margin_buffer : float, optional, default=1.35
        Buffer factor to add around the tissue region when cropping the image.
    remove_background_tissue : bool, optional, default=True
        Whether to remove background tissue from the cropped image.
    tissue_min_size : int, optional, default=5000
        Minimum size of tissue regions to retain when removing background tissue.
    tissue_sigma : float, optional, default=3
        Sigma for Gaussian blur when detecting tissue regions.
    tissue_threshold_rescale_factor : float, optional, default=1.1
        Rescale factor for the Otsu threshold when detecting tissue regions.
    tissue_connectivity : int, optional, default=2
        Connectivity parameter for detecting connected tissue regions.
    preview_alignment : bool, optional, default=True
        Whether to display a preview of the alignment and tissue separation.
    preview_images_pt_size : int, optional, default=1
        Point size for the scatter plot in the preview images.
    preview_images_alpha : float, optional, default=0.5
        Alpha (opacity) for the points in the preview images.
    preview_HE_images : bool, optional, default=True
        Whether to display the cropped H&E images as a preview.

    Returns
    -------
    None
        Saves the separated AnnData objects and optionally displays preview images.

    Raises
    ------
    ValueError
        If the `aspect` parameter cannot be converted to a float.

    Notes
    -----
    - The function separates tissues based on the unique values in the `seperate_column`.
    - Cropped images are saved to the specified `adata_output_folder`.
    - Optionally, background tissue can be removed from the cropped images.
    - Previews of the alignment and the cropped images can be displayed.

    Example
    -------
    >>> seperate_tissues(adata=my_adata, seperate_column='Tissue_ID')
    """
        
    adata_X = adata.copy()
    image = adata_X.uns['Image']

    unique_ids = adata_X.obs.sort_values(seperate_column)[seperate_column].unique(dropna=True)
    
    if preview_HE_images == True:
        HE_images = []
    
    # Check if all unique IDs can be converted to integers
    if all(can_be_converted_to_int(ID) for ID in unique_ids):
        unique_ids = unique_ids.astype(int)

    if preview_alignment == True:
        n_plots = len(unique_ids)
        n_cols = math.ceil(math.sqrt(n_plots))
        n_rows = math.ceil(n_plots / n_cols)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(14,14))
        if n_plots > 1:
            axs = axs.flatten()
        else:
            axs = [axs]
    
    for idx, ID in enumerate(unique_ids):
        
        
            
        adata_X_sub = adata_X[adata_X.obs[seperate_column] == ID]
        
        adata_X_sub_meta = adata_X_sub.obs.copy()

        x_bottom_percentile, x_top_percentile = np.percentile(adata_X_sub_meta["X"], [2.5, 97.5])
        y_bottom_percentile, y_top_percentile = np.percentile(adata_X_sub_meta["Y"], [2.5, 97.5])

        if aspect == "auto": 
            aspect_ratio = np.round((x_top_percentile - x_bottom_percentile) / (y_top_percentile - y_bottom_percentile), decimals=2)
        elif aspect != 'auto':
            try:
                aspect_ratio = float(aspect)
            except ValueError:
                raise ValueError("'aspect' must be a float")
    
        x_range = (x_top_percentile - x_bottom_percentile)*tissue_margin_buffer
        y_range = (y_top_percentile - y_bottom_percentile)*tissue_margin_buffer

        max_dim = int(np.max([x_range, y_range])/2)
        
        x_middle = int(np.mean([x_bottom_percentile, x_top_percentile]))
        y_middle = int(np.mean([y_bottom_percentile, y_top_percentile]))

        y_lower = max(0, y_middle-max_dim)
        y_upper = min(len(image), y_middle+max_dim)

        x_lower = max(0, int(x_middle- np.floor(max_dim*aspect_ratio)))
        x_upper = min(len(image[1]), int(x_middle + np.floor(max_dim*aspect_ratio)))

        HE_image_cropped = image[y_lower:y_upper, x_lower:x_upper]

        adata_X_sub_meta["X"] = adata_X_sub_meta["X"].add(-x_lower) 
        adata_X_sub_meta["Y"] = adata_X_sub_meta["Y"].add(-y_lower) 
        
        if remove_background_tissue == True:
            HE_image_cropped = Remove_Background_Tissue(HE_image_cropped, 
                                                       tissue_min_size, 
                                                         tissue_sigma, 
                                                         tissue_threshold_rescale_factor, 
                                                         tissue_connectivity)

        adata_X_sub.obs = adata_X_sub_meta
        adata_X_sub.uns['Image'] = HE_image_cropped
        scanpy_coords = adata_X_sub.obs[["X", "Y"]].to_numpy()
        adata_X_sub.obsm["spatial"] = scanpy_coords
        
        if preview_HE_images == True:
            HE_images = HE_images + [HE_image_cropped]

        if not os.path.exists(adata_output_folder):
            os.makedirs(adata_output_folder)

        adata_X_sub.write(os.path.join(adata_output_folder, f"{sample_name}_{ID}.h5ad"))

        if preview_alignment == True:
            axs[idx].imshow(rgb2gray(HE_image_cropped), cmap='gray')
            axs[idx].scatter(adata_X_sub_meta["X"], adata_X_sub_meta["Y"], c="r", s=preview_images_pt_size, alpha=preview_images_alpha)
            axs[idx].set_title(f'{seperate_column}={ID}')
    
    if preview_alignment == True:
        # Hide any unused subplots
        for i in range(len(unique_ids), len(axs)):
            axs[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    if preview_HE_images == True:
        n_plots = len(unique_ids)
        n_cols = math.ceil(math.sqrt(n_plots))
        n_rows = math.ceil(n_plots / n_cols)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(14,14))
        if n_plots > 1:
            axs = axs.flatten()
        else:
            axs = [axs]
            
        HE_i = 0
            
        for ID in unique_ids:
            axs[HE_i].imshow(HE_images[HE_i])
            axs[HE_i].set_title(f'{seperate_column}={ID}')
            
            HE_i += 1
        
        for i in range(len(unique_ids), len(axs)):
            axs[i].axis('off')
            
        plt.tight_layout()
        plt.show()





def extract_number(filename):
    match = re.search(r'(\d+)\.h5ad$', filename)
    return int(match.group(1)) if match else float('inf')



def pad_image_to_max_dimensions(image, target_shape):
    current_shape = image.shape
    pad_height = max(0, target_shape[0] - current_shape[0])
    pad_width = max(0, target_shape[1] - current_shape[1])

    # Calculate padding for both sides
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    if image.ndim == 3:
        padded_image = np.pad(image, 
                              ((pad_top, pad_bottom), 
                               (pad_left, pad_right), 
                               (0, 0)), 
                              mode='constant', constant_values=0)
    else:
        padded_image = np.pad(image, 
                              ((pad_top, pad_bottom), 
                               (pad_left, pad_right)), 
                              mode='constant', constant_values=1)
    
    return padded_image, pad_top, pad_left


def align_3D(input_path, 
             output_sample_name = None, 
             output_image_path = None, 
             alignment_pixel_range = 40, 
             alignment_pixel_resolution = 20, 
             rotation_range = 5, 
             rotation_resolution = 1, 
            image_subsample_factor = 1, 
            plot_aligned_ouput = True, 
            pt_size = 1, 
            plot_size = 10):
    
    """
    Aligns a series of 3D histological images and their corresponding AnnData objects. The function reads the images, aligns them based on tissue overlap, and saves the aligned images and AnnData objects.

    Parameters
    ----------
    input_path : str
        Path to the directory containing the .h5ad files with histological images and spatial coordinates.
    output_sample_name : str, optional
        Base name for the output image files. If None, output images will not be saved.
    output_image_path : str, optional
        Directory path to save the aligned images. If None, aligned images will not be saved.
    alignment_pixel_range : int, optional, default=40
        Range of pixels to search for alignment in x and y directions.
    alignment_pixel_resolution : int, optional, default=20
        Resolution of the pixel search grid for alignment.
    rotation_range : int, optional, default=5
        Range of degrees to search for alignment through rotation.
    rotation_resolution : int, optional, default=1
        Resolution of the rotation search grid in degrees.
    image_subsample_factor : int, optional, default=1
        Factor by which to subsample the images for faster alignment.
    plot_aligned_ouput : bool, optional, default=True
        Whether to plot the aligned output.
    pt_size : int, optional, default=1
        Point size for the scatter plot of the aligned output.
    plot_size : int, optional, default=10
        Size of the plot for the aligned output.

    Returns
    -------
    AnnData
        An AnnData object containing the aligned spatial coordinates and the corresponding images.

    Notes
    -----
    - The function assumes that the images are named in a way that their order can be determined from their filenames.
    - The alignment process involves both translation and rotation to maximize the overlap between consecutive tissue images.
    - The function can save the aligned images to a specified directory if `output_image_path` is provided.
    - The function can plot the aligned spatial coordinates if `plot_aligned_output` is set to True.

    Example
    -------
    >>> aligned_adata = align_3D(
    >>>     input_path='input_folder',
    >>>     output_sample_name='aligned_sample',
    >>>     output_image_path='output_folder',
    >>>     alignment_pixel_range=50,
    >>>     alignment_pixel_resolution=10,
    >>>     rotation_range=10,
    >>>     rotation_resolution=2,
    >>>     image_subsample_factor=2,
    >>>     plot_aligned_output=True,
    >>>     pt_size=2,
    >>>     plot_size=12
    >>> )
    """

    sorted_files = [os.path.basename(f) for f in glob.glob(os.path.join(input_path, '*.h5ad')) if os.path.isfile(f)]
    sorted_files.sort(key=extract_number)

    # Extract the detected numbers
    detected_numbers = [extract_number(f) for f in sorted_files]

    print("Sorted files:", sorted_files)
    print("Detected numbers:", detected_numbers)
    
    if image_subsample_factor != 1:
        alignment_pixel_range = alignment_pixel_range//image_subsample_factor
        alignment_pixel_resolution = alignment_pixel_resolution//image_subsample_factor

    
    # Read all images and find the max dimensions
    max_height, max_width = 0, 0
    images = []
    adata_files = []

    for adata_file in sorted_files:
        adata_X_i = sc.read_h5ad(os.path.join(input_path, adata_file))
        image = adata_X_i.uns['Image']
        images.append(image)
        adata_files.append(adata_X_i)

        if image.shape[0] > max_height:
            max_height = image.shape[0]
        if image.shape[1] > max_width:
            max_width = image.shape[1]

    target_shape = (max_height, max_width)


    i = 0
    aligned_images = []

    sorted_files_progress_bar = tqdm(sorted_files, desc="Aligning Images")


    for adata_file in sorted_files_progress_bar: 
        adata_X_i = sc.read_h5ad(os.path.join(input_path, adata_file))
        image = adata_X_i.uns['Image']
        
        
        if i == 0: 
            reference_image_gray = skimage.color.rgb2gray(image)
            reference_tissue = reference_image_gray > threshold_otsu(reference_image_gray)*1.1
            reference_tissue, pad_top, pad_left = pad_image_to_max_dimensions(reference_tissue, target_shape)
            
            
            
            if image_subsample_factor != 1:
                reference_tissue = reference_tissue[::image_subsample_factor, ::image_subsample_factor]
            
            
            image_padded, pad_top, pad_left = pad_image_to_max_dimensions(image, target_shape)
            aligned_images.append(image_padded)

            adata_X_i.obs["X"] = adata_X_i.obs["X"] + pad_left
            adata_X_i.obs["Y"] = adata_X_i.obs["Y"] + pad_top

            adata_X_full = adata_X_i.copy()

            if output_image_path is not None: 
                if not os.path.exists(output_image_path):
                    os.makedirs(output_image_path)
                io.imsave(fname = os.path.join(output_image_path, 
                                               f"{output_sample_name}_{detected_numbers[i]}.png"), 
                          arr=image_padded)

        else: 

            #I might not need the lists below: 
            overlay_percentages_before = []
            overlay_percentages_after = []
            translations = []
            rotations = []


            pre_aligned_image_gray = skimage.color.rgb2gray(image)
            pre_aligned_tissue = pre_aligned_image_gray > threshold_otsu(pre_aligned_image_gray)*1.1

            pre_aligned_tissue, pad_top, pad_left = pad_image_to_max_dimensions(pre_aligned_tissue, target_shape)
            
            if image_subsample_factor != 1:
                pre_aligned_tissue = pre_aligned_tissue[::image_subsample_factor, ::image_subsample_factor]
                



            initial_intersection = np.logical_and(reference_tissue, pre_aligned_tissue)
            initial_union = np.logical_or(reference_tissue, pre_aligned_tissue)
            initial_overlap_percentage = (np.sum(initial_intersection) / np.sum(initial_union)) * 100
            overlay_percentages_before.append(initial_overlap_percentage)

            best_translation = (0, 0)
            best_rotation = 0
            max_overlap_percentage = overlay_percentages_before
            


            for dx in range(-alignment_pixel_range, alignment_pixel_range, alignment_pixel_resolution):
                Aligning_tissue_x = np.roll(pre_aligned_tissue, dx, axis=1)

                for dy in range(-alignment_pixel_range, alignment_pixel_range, alignment_pixel_resolution):
                    Aligning_tissue_xy = np.roll(Aligning_tissue_x, dy, axis=0)

                    for angle in range(-rotation_range, rotation_range, rotation_resolution):

                        rotated_translated_image = transform.rotate(Aligning_tissue_xy, angle, mode='edge', preserve_range=True)

                        intersection = np.logical_and(reference_tissue, rotated_translated_image)
                        union = np.logical_or(reference_tissue, rotated_translated_image)
                        overlap_percentage = (np.sum(intersection) / np.sum(union)) * 100

                        if overlap_percentage > max_overlap_percentage:
                            max_overlap_percentage = overlap_percentage
                            best_translation = (dx, dy)
                            best_rotation = angle
                            

            
            #Applying best tranformation values to image
            image_padded, pad_top, pad_left = pad_image_to_max_dimensions(image, target_shape)
            translated_image_color = np.roll(image_padded, best_translation[0], axis=1)
            translated_image_color = np.roll(translated_image_color, best_translation[1], axis=0)
            rotated_translated_image_color = transform.rotate(translated_image_color, best_rotation, mode='edge', preserve_range=True)
            
            aligned_images.append(rotated_translated_image_color)

            if output_image_path is not None: 
                if not os.path.exists(output_image_path):
                    os.makedirs(output_image_path)
                io.imsave(fname = os.path.join(output_image_path, 
                                               f"{output_sample_name}_{detected_numbers[i]}.png"), 
                          arr=rotated_translated_image_color)

            #Applying best tranformation values to adata
            if image_subsample_factor != 1:
                adata_X_i.obs["Y"] = adata_X_i.obs["Y"].add(best_translation[1]*image_subsample_factor)
                adata_X_i.obs["X"] = adata_X_i.obs["X"].add(best_translation[0]*image_subsample_factor)
            else: 
                adata_X_i.obs["Y"] = adata_X_i.obs["Y"].add(best_translation[1])
                adata_X_i.obs["X"] = adata_X_i.obs["X"].add(best_translation[0])


            
            

            x_bottom_percentile, x_top_percentile = np.percentile(adata_X_i.obs["X"], [5, 95])
            y_bottom_percentile, y_top_percentile = np.percentile(adata_X_i.obs["Y"], [5, 95])

            x_middle = np.mean([x_bottom_percentile, x_top_percentile])
            y_middle = np.mean([y_bottom_percentile, y_top_percentile])

            # Adjusting coordinates to the new origin
            adata_X_i.obs['X'] = adata_X_i.obs['X'] - x_middle
            adata_X_i.obs['Y'] = adata_X_i.obs['Y'] - y_middle

            # Rotation angle in degrees and conversion to radians
            angle_radians = np.radians(best_rotation)

            # Rotation matrix
            rotation_matrix = np.array([
                [np.cos(angle_radians), -np.sin(angle_radians)],
                [np.sin(angle_radians), np.cos(angle_radians)]
            ])


            # Apply rotation to adjusted (X, Y) pairs
            rotated_coordinates = np.dot(adata_X_i.obs[['X', 'Y']], rotation_matrix)

            # Update DataFrame with rotated coordinates
            adata_X_i.obs['X'] = rotated_coordinates[:, 0] + x_middle  # Adjust back after rotation
            adata_X_i.obs['Y'] = rotated_coordinates[:, 1] + y_middle  # Adjust back after rotation 
            
            adata_X_i.obs["X"] = adata_X_i.obs["X"] + pad_left
            adata_X_i.obs["Y"] = adata_X_i.obs["Y"] + pad_top
          
            scanpy_coords = adata_X_i.obs[["X", "Y"]].to_numpy()
            adata_X_i.obsm["spatial"] = scanpy_coords
            

            adata_X_full = adata_X_full.concatenate(adata_X_i)
            gc.collect()

            #Setting new reference image: 
            pre_aligned_tissue = np.roll(pre_aligned_tissue, best_translation[0], axis=1)
            pre_aligned_tissue = np.roll(pre_aligned_tissue, best_translation[1], axis=0)
            reference_tissue = transform.rotate(pre_aligned_tissue, best_rotation, mode='edge', preserve_range=True)


        i+=1
    
    if plot_aligned_ouput == True:
        meta = adata_X_full.obs.sort_values("Z", ascending=False)
        fig, ax = plt.subplots(figsize=(plot_size, plot_size))
        ax.imshow(np.ones((max_height, max_width)), cmap='gray')
        scatter = ax.scatter(meta["X"], 
                             meta["Y"], 
                             c=meta["Z"], 
                             s=pt_size, 
                             edgecolor='none')
        plt.show()

    adata_X_full.uns['Image'] = aligned_images
    return adata_X_full
