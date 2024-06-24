import warnings
from math import floor
import numpy as np
import pandas as pd
import skimage

warnings.filterwarnings('ignore')

def check_arrayseq_signature(input_signature, accepted_signature = None, accepted_signatures = None): 
    if accepted_signature is not None: 
        if input_signature == accepted_signature:
            pass
        elif accepted_signature == "Auto Aligned" or accepted_signature == "Manually Aligned": 
            raise ValueError("Unexpected adata signature. Expected auto-aligned or manually aligned adata. Check that the adata object was generated from the ___ or ___ funtions.")
        elif accepted_signature == "Full ArraySeq Object": 
            raise ValueError("Unexpected adata signature. Expected a full ArraySeq object. Check that the adata object was generated from the ___ function.")
        elif accepted_signature == "Separated ArraySeq Object": 
            raise ValueError("Unexpected adata signature. Expected a full ArraySeq object. Check that the adata object was generated from the ___ function.")
        elif accepted_signature == "3D ArraySeq Object": 
            raise ValueError("Unexpected adata signature. Expected a 3D ArraySeq object. Check that the adata object was generated from the ___ function.")
    elif accepted_signatures is not None: 
        if input_signature in accepted_signatures:
            pass
        else: 
            raise ValueError(f"Unexpected adata signature. This function expects an adata object with the following signatures: {accepted_signatures}.")


def under_tissue_function(x, y, HE_Width, HE_Height, tissue_mask):
    if x >=0 and y >= 0 and x < HE_Width and y < HE_Height: 
        y_pixel = int(floor(y))
        x_pixel = int(floor(x))
        truth_value = tissue_mask[y_pixel, x_pixel] == True
    else:
        truth_value = False
            
    return truth_value



def cavity_function(x, y, HE_Width, HE_Height, red_channel, path_to_region_selection_image):
    if x >=0 and y >= 0 and x <= HE_Width and y <= HE_Height: 
        y_pixel = int(floor(y))
        x_pixel = int(floor(x))
        if type(path_to_region_selection_image) == str:
            if red_channel[y_pixel, x_pixel] > 150:
                truth_value = True
            else:
                truth_value = False
        else: 
            truth_value = False
    else:
        truth_value = False
            
    return truth_value




def region_selection_function(x, y, HE_Width, HE_Height, green_channel, path_to_region_selection_image):
    if type(path_to_region_selection_image) == str:
        if x >=0 and y >= 0 and x <= HE_Width and y <= HE_Height: 
            y_pixel = int(floor(y))
            x_pixel = int(floor(x))
            if green_channel[y_pixel, x_pixel] > 150:
                truth_value = True
            else:
                truth_value = False
        else:
            truth_value = False
    else:
        truth_value = False
            
    return truth_value


def create_arrayseq(adata, 
                    HE_Image,
                    tissue_mask, 
                    path_to_region_selection_image = None, 
                    labeled_mask = None):  
    
    """
    Create a spatial transcriptomics (ST) object with additional metadata, including tissue and region selection.

    This function processes a given AnnData object containing ST data, aligning it to a histology image (H&E Image) 
    and applying various masks and selections based on tissue and region data. The result is a refined AnnData object 
    with updated metadata.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing spatial transcriptomics data.
    HE_Image : np.ndarray
        A 2D or 3D numpy array representing the histology image (H&E Image).
    tissue_mask : np.ndarray
        A binary mask indicating the tissue regions in the histology image.
    path_to_region_selection_image : str, optional
        Path to an image file used for region selection. This image should contain specific color channels 
        (red and green) to identify cavities and selected regions, respectively.
    labeled_mask : np.ndarray, optional
        A labeled mask indicating different tissue regions. Each label corresponds to a different tissue type.

    Returns
    -------
    AnnData
        The processed AnnData object with updated metadata, including tissue and region information.

    Notes
    -----
    - The function verifies the alignment signature of the input AnnData object, ensuring it has been 
      aligned either automatically or manually.
    - It processes the histology image and extracts relevant spatial information, mapping it to the spatial 
      coordinates in the AnnData object.
    - If a region selection image is provided, it uses the red channel to identify cavities and the green 
      channel to select specific regions.
    - The function removes data points that do not fall under the tissue mask or are identified as cavities.
    - If a labeled mask is provided, it assigns labels to the data points based on their spatial coordinates.

    Example
    -------
    >>> adata_processed = create_arrayseq(
    >>>     adata=adata,
    >>>     HE_Image=he_image,
    >>>     tissue_mask=tissue_mask,
    >>>     path_to_region_selection_image="path/to/region_selection_image.png",
    >>>     labeled_mask=labeled_mask
    >>> )
    """
    
    check_arrayseq_signature(adata.uns.get("Signature"), accepted_signatures = ["Auto Aligned", "Manually Aligned"])
    
    HE_Height = len(HE_Image)
    HE_Width = len(HE_Image[1])

    adata_x = adata.copy()

    adata_x.uns['Image'] = HE_Image
    
    if type(path_to_region_selection_image) == str:
        selection_image = skimage.io.imread(path_to_region_selection_image)[:,:,:3]
        red_channel = selection_image[:, :, 0]
        green_channel = selection_image[:, :, 1]
        

        cavity_bool_list = [cavity_function(x, y, HE_Width, HE_Height, red_channel = red_channel, path_to_region_selection_image = path_to_region_selection_image) for x, y in zip(adata_x.obs['X'], adata_x.obs['Y'])]
        adata_x.obs["Cavity"] = cavity_bool_list

        selected_bool_list = [region_selection_function(x, y, HE_Width, HE_Height, green_channel, path_to_region_selection_image) for x, y in zip(adata_x.obs['X'], adata_x.obs['Y'])]
        adata_x.obs["Selected"] = selected_bool_list
        
        under_tissue_bool_list = [under_tissue_function(x, y, HE_Width, HE_Height, tissue_mask) for x, y in zip(adata_x.obs['X'], adata_x.obs['Y'])]
        adata_x = adata_x[under_tissue_bool_list]
        
    else: 
        under_tissue_bool_list = [under_tissue_function(x, y, HE_Width, HE_Height, tissue_mask) for x, y in zip(adata_x.obs['X'], adata_x.obs['Y'])]
        adata_x = adata_x[under_tissue_bool_list]
        
        cavity_bool_list = [cavity_function(x, y, HE_Width, HE_Height, red_channel = None, path_to_region_selection_image = None) for x, y in zip(adata_x.obs['X'], adata_x.obs['Y'])]
        adata_x.obs["Cavity"] = cavity_bool_list
        
        
    if labeled_mask is not None:
        adata_x.obs['Tissue_Label'] = labeled_mask[(adata_x.obs['Y'].apply(lambda x: int(np.floor(x))), 
                                                   adata_x.obs['X'].apply(lambda x: int(np.floor(x))))]
        
    adata_x = adata_x[adata_x.obs["Cavity"] == False]
    del adata_x.obs["Cavity"]
    
    adata_x.uns["Signature"] = "Full ArraySeq Object"
    scanpy_coords = adata_x.obs[["X", "Y"]].to_numpy()
    adata_x.obsm["spatial"] = scanpy_coords
    
    return adata_x


def can_be_converted_to_int(value):
    if pd.isna(value):
        return True
    return np.isclose(value, int(value))

def convert_columns_to_int_if_possible(df, exclude_col):
    for col in df.columns:
        if col != exclude_col and np.issubdtype(df[col].dtype, np.number):
            if all(can_be_converted_to_int(value) for value in df[col]):
                df[col] = df[col].apply(lambda x: int(x) if not pd.isna(x) else x)
    return df

def add_metadata(adata, 
                 metadata,
                 meta_join_col="Tissue_Label", 
                 adata_join_col="Tissue_Label"):
    
    """
    Add metadata to an AnnData object by joining on specified columns.

    This function enhances an AnnData object containing spatial transcriptomics (ST) data by merging it 
    with additional metadata. The join is performed on specified columns from the AnnData object and 
    the metadata DataFrame, allowing for flexible integration of supplementary information.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing spatial transcriptomics data. The object should have a specific 
        alignment signature in its `.uns` attribute.
    metadata : pd.DataFrame
        A pandas DataFrame containing additional metadata to be added to the AnnData object.
    meta_join_col : str, optional
        The column name in the metadata DataFrame used for joining. Default is "Tissue_Label".
    adata_join_col : str, optional
        The column name in the AnnData object's `.obs` DataFrame used for joining. Default is "Tissue_Label".

    Returns
    -------
    AnnData
        The modified AnnData object with the additional metadata merged into its `.obs` attribute.

    Notes
    -----
    - The function first checks the alignment signature of the input AnnData object to ensure it is a 
      compatible ArraySeq object.
    - It creates copies of the input AnnData and metadata to avoid modifying the original objects.
    - The function verifies the data types of the joining columns and attempts to cast them to the same type 
      if they differ, providing detailed messages about the process.
    - It converts columns to integers where possible to facilitate accurate joining.
    - The join operation merges the metadata into the AnnData object's `.obs` DataFrame based on the specified 
      columns.

    Example
    -------
    >>> adata_with_metadata = add_metadata(
    >>>     adata=adata,
    >>>     metadata=metadata_df,
    >>>     meta_join_col="Tissue_Label",
    >>>     adata_join_col="Tissue_Label"
    >>> )
    """
    
    # Check the signature of the arrayseq object in adata.uns
    check_arrayseq_signature(adata.uns.get("Signature"), accepted_signatures=["Full ArraySeq Object", "Separated ArraySeq Object", "3D ArraySeq Object"])

    # Copy the input data to avoid modifying the original objects
    adata_X = adata.copy()
    metadata_X = metadata.copy()

    # Check if the types of the joining columns are different
    if adata.obs[adata_join_col].dtype != metadata_X[meta_join_col].dtype:
        print(f"The metadata and adata joining columns are not the same type. Attempting to type-cast metadata column '{meta_join_col}' from "
              f"{metadata_X[meta_join_col].dtype} to {adata.obs[adata_join_col].dtype}.")

        # Attempt to type-cast metadata column to the type of adata column
        try:
            metadata_X[meta_join_col] = metadata_X[meta_join_col].astype(adata.obs[adata_join_col].dtype)
            print(f"Successfully casted metadata column '{meta_join_col}' to {adata.obs[adata_join_col].dtype}.")
        except ValueError as e:
            print(f"Error casting metadata column '{meta_join_col}': {e}")
            raise

    # Convert columns to integers where possible
    metadata_X = convert_columns_to_int_if_possible(metadata_X, meta_join_col)

    # Join the metadata with adata.obs on the specified columns
    adata_X.obs = adata_X.obs.join(metadata_X.set_index(meta_join_col), on=adata_join_col)

    return adata_X


def mirror_image_and_coordinates(adata, axis='x'):
    
    """
    Mirror an image and its associated coordinates in an AnnData object along a specified axis.

    This function mirrors the spatial transcriptomics (ST) image and the corresponding coordinates 
    in an AnnData object along the specified axis ('x' or 'y'). It updates both the image and the 
    spatial coordinates to reflect the mirroring transformation.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing spatial transcriptomics data, including an image in the `.uns` attribute.
    axis : str, optional
        The axis along which to mirror the image and coordinates. Accepted values are 'x' and 'y'. 
        Default is 'x'.

    Returns
    -------
    AnnData
        The modified AnnData object with the mirrored image and updated coordinates.

    Raises
    ------
    ValueError
        If an invalid axis is specified.

    Notes
    -----
    - The function first checks the alignment signature of the input AnnData object to ensure it is a 
      compatible ArraySeq object.
    - It creates a copy of the input AnnData object to avoid modifying the original data.
    - The image and coordinates are mirrored along the specified axis:
        - For the 'x' axis, the image is flipped horizontally, and the x-coordinates are adjusted.
        - For the 'y' axis, the image is flipped vertically, and the y-coordinates are adjusted.
    - The mirrored image is updated in the `.uns` attribute of the AnnData object.

    Example
    -------
    >>> mirrored_adata = mirror_image_and_coordinates(adata=adata, axis='x')
    """

    check_arrayseq_signature(adata.uns.get("Signature"), accepted_signatures=["Full ArraySeq Object", "Separated ArraySeq Object"])

    adata_X = adata.copy()
    
    image = adata_X.uns['Image']

    if axis == 'x':
        mirrored_image = np.flip(image, axis=1)
        max_x_image = image.shape[1]
        adata_X.obs['X'] = max_x_image - adata_X.obs['X']
    elif axis == 'y':
        mirrored_image = np.flip(image, axis=0)
        max_y_image = image.shape[0]
        adata_X.obs['Y'] = max_y_image - adata_X.obs['Y']
    else:
        raise ValueError("Invalid axis specified. Use 'x' or 'y'.")
    
    adata_X.uns['Image'] = mirrored_image
                     
    scanpy_coords = adata_X.obs[["X", "Y"]].to_numpy()
    adata_X.obsm["spatial"] = scanpy_coords

    return adata_X


def rotate_image_and_coordinates(adata, angle=90):
    """
    Rotate an image and its associated coordinates in an AnnData object by a specified angle.

    This function rotates the spatial transcriptomics (ST) image and the corresponding coordinates 
    in an AnnData object by the specified angle (90, 180, or 270 degrees). It updates both the image 
    and the spatial coordinates to reflect the rotation transformation.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing spatial transcriptomics data, including an image in the `.uns` attribute.
    angle : int, optional
        The angle by which to rotate the image and coordinates. Accepted values are 90, 180, and 270 degrees. 
        Default is 90.

    Returns
    -------
    AnnData
        The modified AnnData object with the rotated image and updated coordinates.

    Raises
    ------
    ValueError
        If an invalid angle is specified.

    Notes
    -----
    - The function first checks the alignment signature of the input AnnData object to ensure it is a 
      compatible ArraySeq object.
    - It creates a copy of the input AnnData object to avoid modifying the original data.
    - The image and coordinates are rotated by the specified angle:
        - For 90 degrees, the image is rotated counterclockwise and the coordinates are adjusted accordingly.
        - For 180 degrees, the image is rotated by 180 degrees and the coordinates are adjusted accordingly.
        - For 270 degrees, the image is rotated clockwise and the coordinates are adjusted accordingly.
    - The rotated image is updated in the `.uns` attribute of the AnnData object.

    Example
    -------
    >>> rotated_adata = rotate_image_and_coordinates(adata=adata, angle=90)
    """

    check_arrayseq_signature(adata.uns.get("Signature"), accepted_signatures=["Full ArraySeq Object", "Separated ArraySeq Object"])

    if angle not in [90, 180, 270]:
        raise ValueError("Angle must be one of 90, 180, or 270.")
    
    adata_X = adata.copy()
    
    image = adata_X.uns['Image']

    rotated_image = np.rot90(image, k=angle // 90)
    
    if angle == 90:
        max_y_image = image.shape[0]
        adata_X.obs['X'], adata_X.obs['Y'] = adata_X.obs['Y'], max_y_image - adata_X.obs['X']
    elif angle == 180:
        max_x_image = image.shape[1]
        max_y_image = image.shape[0]
        adata_X.obs['X'] = max_x_image - adata_X.obs['X']
        adata_X.obs['Y'] = max_y_image - adata_X.obs['Y']
    elif angle == 270:
        max_x_image = image.shape[1]
        adata_X.obs['X'], adata_X.obs['Y'] = max_x_image - adata_X.obs['Y'], adata_X.obs['X']

    adata_X.uns['Image'] = rotated_image

    scanpy_coords = adata_X.obs[["X", "Y"]].to_numpy()
    adata_X.obsm["spatial"] = scanpy_coords
    
    return adata_X


