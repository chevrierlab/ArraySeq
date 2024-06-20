import warnings
import numpy as np
import matplotlib.pyplot as plt
import skimage 
import os
from skimage import io, measure, color
from scipy import ndimage


warnings.filterwarnings('ignore')

def convert_to_grayscale(image):
    """Converts an RGB image to grayscale if it is in RGB format."""
    if image.ndim == 3 and image.shape[2] == 3:
        # Image is in RGB format
        return color.rgb2gray(image)
    elif image.ndim == 2:
        # Image is already in grayscale format
        return image
    else:
        raise ValueError("Input image must be either an RGB or grayscale image.")




def preprocess_histology_image(
    image_path,
    sample_name = None,
    output_folder_path = None,
    intensity_rescale_factor = 1.1,
    show_image = True, 
    save_images = True, 
    image_size = 6):

    """
    Preprocess a histology image by rescaling its intensity and optionally saving and displaying the result.

    This function reads a histology image, converts it to grayscale, applies intensity rescaling, and optionally saves 
    and displays the processed image. The function also ensures the necessary directories exist for saving the images.

    Parameters
    ----------
    image_path : str
        Path to the input histology image file.
    sample_name : str, optional
        Name of the sample to be used in the output file names. Required if 'save_images' is True.
    output_folder_path : str, optional
        Path to the folder where the processed images will be saved. Required if 'save_images' is True.
    intensity_rescale_factor : float, optional, default=1.1
        Factor by which to rescale the intensity of the image.
    show_image : bool, optional, default=True
        Whether to display the processed image.
    save_images : bool, optional, default=True
        Whether to save the processed images.
    image_size : int, optional, default=6
        Size of the displayed image (for matplotlib).

    Returns
    -------
    rescaled_rgb : ndarray
        The processed RGB image with rescaled intensity.

    Raises
    ------
    ValueError
        If 'sample_name' and 'output_folder_path' are not provided when 'save_images' is True.
    
    IOError
        If there is an issue with reading or writing the image files.
    
    Notes
    -----
    - The function reads the input image and ensures it is in RGB format if it has more than 3 channels.
    - The grayscale conversion and intensity rescaling are performed using the Otsu threshold.
    - The function ensures that the 'output_folder_path' exists before saving images.
    - If 'save_images' is True, the processed images are saved as PNG files in the specified output folder.
    - The function can display the processed image using matplotlib if 'show_image' is set to True.

    Example
    -------
    >>> processed_image = preprocess_histology_image(
    >>>     image_path='path/to/image.png',
    >>>     sample_name='sample1',
    >>>     output_folder_path='path/to/output'
    >>> )
    """
        
    if None in (sample_name, output_folder_path) and save_images:
        raise ValueError("Both 'sample_name' and 'output_folder_path' are required when 'save_images' is True.")


    image = skimage.io.imread(image_path)
    if image.shape[-1] >= 3:
        image = image[..., :3]
    grayscale = image if image.ndim == 2 else skimage.color.rgb2gray(image)

    thresh = skimage.filters.threshold_otsu(grayscale)
    #include a check to make sure it falls within limits here and for the H&E image:
    rescaled_grayscale = skimage.exposure.rescale_intensity(grayscale, in_range=(0, thresh*intensity_rescale_factor))


    max_intensity = image.max()
    rescaled_rgb = np.zeros_like(image)
    for i in range(3):
        rescaled_rgb[:,:,i] = skimage.exposure.rescale_intensity(image[:,:,i], in_range=(0, max_intensity*thresh*intensity_rescale_factor))

    #add a try except statement here:
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        
        
    if save_images:
        io.imsave(fname = os.path.join(output_folder_path, f"{sample_name}_HE.png"), arr=rescaled_rgb)
        io.imsave(fname = os.path.join(output_folder_path, f"{sample_name}_BW.png"), arr=skimage.img_as_ubyte(rescaled_grayscale))

    if show_image == True: 
        plt.figure(figsize=(image_size, image_size))
        plt.imshow(rescaled_rgb)
        plt.axis('off')
        plt.title("H&E")
        
        plt.show()
        
    
    return rescaled_rgb



def detect_tissue(image, 
                sigma = 1, 
                manual_threshold_intensity = None, 
                object_connectivity = 2,
                min_tissue_size = 0.001,
                min_hole_size = 0.00005,
                show_plots = True, 
                label_multiple_tissues = False):
    
    """
    Detects tissue regions in a histology image by applying thresholding, object removal, and hole filling.

    This function processes a histology image to identify and segment tissue regions. The process involves converting
    the image to grayscale, applying Gaussian blurring, thresholding, and removing small objects and holes. Optionally,
    it can label multiple tissue regions and display intermediate results.

    Parameters
    ----------
    image : ndarray
        Input histology image.
    sigma : float, optional, default=1
        Standard deviation for Gaussian blur.
    manual_threshold_intensity : float or None, optional, default=None
        Manual threshold intensity for tissue detection. If None, Otsu's method is used.
        Must be a float between 0 and 1, inclusive.
    object_connectivity : int, optional, default=2
        Connectivity for identifying small objects to remove.
    min_tissue_size : float, optional, default=0.001
        Minimum size of tissue regions to keep, as a fraction of the image size.
        Must be a float between 0 and 1, inclusive.
    min_hole_size : float, optional, default=0.00005
        Minimum size of holes to remove, as a fraction of the image size.
        Must be a float between 0 and 1, inclusive.
    show_plots : bool, optional, default=True
        Whether to display the processing steps and final result.
    label_multiple_tissues : bool, optional, default=False
        Whether to label distinct tissue in the output. Should only be set to True if multiple tissues are present in both the image and matrix data.

    Returns
    -------
    tissue_holes_removed : ndarray
        Binary mask of detected tissue with holes removed.
    labeled_image : ndarray, optional
        Labeled image of detected tissue regions, returned only if 'label_multiple_tissues' is True.

    Raises
    ------
    ValueError
        If 'manual_threshold_intensity', 'min_tissue_size', or 'min_hole_size' are not within the range [0, 1].

    Notes
    -----
    - The function applies Gaussian blurring to the grayscale image before thresholding.
    - Small objects and holes are removed based on the specified minimum sizes.
    - If 'show_plots' is True, intermediate steps and results are displayed using matplotlib.
    - If 'label_multiple_tissues' is True, each detected tissue region is labeled and displayed with a color map.

    Example
    -------
    >>> tissue_mask, labeled_tissues = detect_tissue(
    >>>     image=HE_image,
    >>>     label_multiple_tissues=True
    >>> )
    """
    
    grayscale_image = convert_to_grayscale(image)

    blurred_image = skimage.filters.gaussian(grayscale_image, sigma=sigma)


    if manual_threshold_intensity is None:
        threshold = skimage.filters.threshold_otsu(blurred_image)*1.1
    # Ensure the threshold is either 'auto' or a float
    elif manual_threshold_intensity != 'auto':
        try:
            threshold = float(manual_threshold_intensity)
        except ValueError:
            raise ValueError("'manual_threshold_intensity' must be a float")
        if not (0 <= manual_threshold_intensity <= 1):
            raise ValueError("'manual_threshold_intensity' must be between 0 and 1, inclusive.")
        
        threshold = float(manual_threshold_intensity)
    
    
    binary_mask = blurred_image < threshold



    try:
        float(min_tissue_size)
    except ValueError:
        raise ValueError("'min_tissue_size' must be a float")
    if not (0 <= min_hole_size <= 1):
        raise ValueError("'min_tissue_size' must be between 0 and 1, inclusive.")
    
    min_size = image.size * min_tissue_size
    
    object_mask = skimage.morphology.remove_small_objects(binary_mask,min_size, connectivity = object_connectivity)

    holes_filled = ndimage.morphology.binary_fill_holes(object_mask)
    
    try:
        float(min_hole_size)
    except ValueError:
        raise ValueError("'min_hole_size' must be a float")
    if not (0 <= min_hole_size <= 1):
        raise ValueError("'min_hole_size' must be between 0 and 1, inclusive.")
    
    
    holes = holes_filled.astype(np.int8) - object_mask.astype(np.int8)
    
    selected_holes = skimage.morphology.remove_small_objects(holes > 0, min_size = min_hole_size)

    tissue_holes_removed = holes_filled.astype(np.int8) - selected_holes.astype(np.int8)

    if label_multiple_tissues == True:
        labeled_image = skimage.measure.label(tissue_holes_removed)
        regions = measure.regionprops(labeled_image)

    
    if show_plots == True:
        fig = plt.figure()

        fig.add_subplot(1,4,1)
        plt.imshow(grayscale_image, cmap = "gray")
        plt.title("H&E \n Grayscale", fontsize=10)
        plt.axis('off')

        fig.add_subplot(1,4,2)
        plt.imshow(holes_filled)
        plt.title("Detected Tissue \n(Filled in)", fontsize=10)
        plt.axis('off')           

        fig.add_subplot(1,4,3)
        plt.imshow(tissue_holes_removed)
        plt.title("Detected Tissue \n(Holes Removed)", fontsize=10)
        plt.axis('off')
        
        
        if label_multiple_tissues == True: 
        
            fig.add_subplot(1,4,4)
            plt.imshow(labeled_image)
            plt.imshow(labeled_image, cmap='nipy_spectral')  # Using a spectral colormap to distinguish labels

            # Annotate each region with its label
            for region in regions:
                # Get the coordinates of the centroid
                y, x = region.centroid

                # Use matplotlib to annotate the centroid of each region
                plt.annotate(str(region.label), (x, y), color='white', weight='bold', fontsize=15, ha='center')

            plt.colorbar()  # Optionally add a colorbar
            plt.axis('off')  # Turn off the axis
            plt.title("Tissues Labeled", fontsize=10)
            plt.show()


            plt.tight_layout()

            return tissue_holes_removed, labeled_image
        else: 
            plt.tight_layout()

            return tissue_holes_removed

        
