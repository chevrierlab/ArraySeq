
import warnings
import numpy as np
import pandas as pd
import os
import gzip
from wand.image import Image
import svgwrite
import pkg_resources

warnings.filterwarnings('ignore')

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
    
#Read in STARsolo output and write svg file and scanpy object
def write_svg(adata, sample_name, output_folder_path, Barcode_coord_file):
    adata_X = adata.copy()

    adata_X = map_barcodes(adata_X)
    adata_X.obs["UMI"] = np.concatenate(adata_X.X.sum(axis=1).tolist()).flat


    #Scale color intensity of each spot based on UMI/Spot
    color = adata.obs["UMI"].divide(adata.obs["UMI"].max()).multiply(255)

    #Write SVG by making a circle graphic for each spot on the array colored by UMI/Spot
    dwg = svgwrite.Drawing(os.path.join(output_folder_path, f"{sample_name}_Spatial_UMI.svg"))
    for i in range(0,len(adata.obs)):
        dwg.add(dwg.circle(center=(adata.obs["X"], adata.obs["Y"]), r=1, stroke=svgwrite.rgb(round(color[i]), 0, 0, '%'), fill=svgwrite.rgb(round(color[i]), 0, 0, '%')))
    dwg.save()


#Read in SVG object and convert to PNG
def write_png(output_folder_path, sample_name, dpi):
    with Image(filename=os.path.join(output_folder_path, f"{sample_name}_Spatial_UMI.svg"), 
               background="transparent", 
               resolution=dpi) as image:
        image.save(filename=os.path.join(output_folder_path, f"{sample_name}_Spatial_UMI.png"))
    os.remove(os.path.join(output_folder_path, f"{sample_name}_Spatial_UMI.svg"))


def generate_manual_ST_image(adata, 
                             sample_name = None,
                             output_folder_path = "./ST_Spot_Image/",
                             Barcode_coord_file = None, 
                             dpi = 225): 
    """
    Generate a spatial transcriptomics (ST) image from STARsolo output and write it as an SVG and PNG file.

    This function reads spatial transcriptomics data, maps barcodes to coordinates, and scales color intensity based on UMI (Unique Molecular Identifier) counts per spot. The result is saved as an SVG file and then converted to a PNG file.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing spatial transcriptomics data with 'X' and 'Y' coordinates and UMI counts.
    sample_name : str, optional
        Name of the sample. This is used as the base name for the output files. Default is None.
    output_folder_path : str, optional
        Path to the folder where the output files will be saved. Default is "./ST_Spot_Image/".
    Barcode_coord_file : str, optional
        Path to the file containing barcode coordinates. Default is None.
    dpi : int, optional
        Resolution for the PNG output file. Default is 225.

    Notes
    -----
    - The function maps barcodes to coordinates using the `Barcode_coord_file`.
    - The UMI counts are used to scale the color intensity of each spot on the array.
    - An SVG file is generated by creating a circle graphic for each spot, colored based on UMI counts.
    - The SVG file is then converted to a PNG file.

    Example
    -------
    >>> generate_manual_ST_image(
    >>>     adata=adata,
    >>>     sample_name="sample1",
    >>>     output_folder_path="./ST_Spot_Image/",
    >>>     Barcode_coord_file="path/to/barcode_coords.csv",
    >>>     dpi=300
    >>> )
    """

    print("Writing Spatial UMI svg file")
    write_svg(adata, sample_name, output_folder_path, Barcode_coord_file)
    print("Converting to png")
    write_png(output_folder_path, sample_name, dpi)
    print("Done")



def maunal_align(adata, 
                    height_HE, 
                    width_HE, 
                    height_ST,
                    width_ST, 
                    HE_x, 
                    HE_y, 
                    ST_x, 
                    ST_y, 
                    path_to_barcode_positions = None):
    """
    Manually align spatial transcriptomics (ST) data to histology image coordinates.

    This function manually aligns ST data to a histology image by adjusting the coordinates of barcodes based on the specified positions and dimensions of the histology image and ST array.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing spatial transcriptomics data.
    height_HE : float
        Height of the histology image in pixels.
    width_HE : float
        Width of the histology image in pixels.
    height_ST : float
        Height of the spatial transcriptomics (ST) array in pixels.
    width_ST : float
        Width of the spatial transcriptomics (ST) array in pixels.
    HE_x : float
        X-coordinate of the center of the histology image.
    HE_y : float
        Y-coordinate of the center of the histology image.
    ST_x : float
        X-coordinate of the center of the ST array.
    ST_y : float
        Y-coordinate of the center of the ST array.
    path_to_barcode_positions : str
        Path to the CSV file containing barcodes and coordinates.

    Returns
    -------
    AnnData
        The AnnData object with updated spatial coordinates for the barcodes after alignment.

    Notes
    -----
    - The function calculates the shifts required to align the ST array to the histology image based on their respective dimensions and specified center positions.
    - It adjusts the coordinates of the barcodes accordingly and updates the AnnData object.

    Example
    -------
    >>> adata_aligned = maunal_align(
    >>>     adata,
    >>>     height_HE=5000,
    >>>     width_HE=4000,
    >>>     height_ST=6000,
    >>>     width_ST=4500,
    >>>     HE_x=2500,
    >>>     HE_y=2000,
    >>>     ST_x=3000,
    >>>     ST_y=3500
    >>> )
    """
    
    adata_x = adata.copy()
    adata_x = map_barcodes(adata_x, path_to_barcode_positions)
    
    HE_position_x = round(HE_x - (width_HE/2))
    HE_position_y = round(HE_y - (height_HE/2))
    ST_position_x = round(ST_x - (width_ST/2))
    ST_position_y = round(ST_y - (height_ST/2))
    

    ST_shift_x = (ST_position_x - HE_position_x)
    ST_shift_y = (ST_position_y - HE_position_y)
    coords = pd.read_csv(path_to_barcode_positions)
    coords["X"] = coords["X"].multiply(width_ST/coords["X"].max()).add(ST_shift_x)
    coords["Y"] = coords["Y"].multiply(height_ST/coords["Y"].max()).add(ST_shift_y)
    adata_x.obs = adata_x.obs.join(coords.set_index("Barcode"))
    
    return adata_x

