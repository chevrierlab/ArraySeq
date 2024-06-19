
import warnings
import numpy as np
import pandas as pd
import os
from wand.image import Image
import svgwrite
import pkg_resources

warnings.filterwarnings('ignore')

def map_barcodes(adata, Barcode_coord_file=None):
    adata_x = adata.copy()

    # Load the barcode coordinate file from within the package if not provided
    if Barcode_coord_file is None:
        Barcode_coord_file = pkg_resources.resource_filename('arrayseq', 'data/ArraySeq_n12_mapping.csv')

    coords = pd.read_csv(Barcode_coord_file, index_col=0)
    adata_x.obs["Barcode"] = adata_x.obs.index
    adata_x.obs = adata_x.obs.join(coords.set_index("Barcode"), on="Barcode")

    adata_x.obs["X"], adata_x.obs["Y"] = adata_x.obs["X"] * 100, adata_x.obs["Y"] * 100

    return adata_x

def write_svg(adata, sample_name, output_folder_path, Barcode_coord_file):
    adata_X = adata.copy()
    adata_X = map_barcodes(adata_X)
    adata_X.obs["UMI"] = np.concatenate(adata_X.X.values)

    # Generate SVG
    dwg = svgwrite.Drawing(os.path.join(output_folder_path, f"{sample_name}.svg"), profile='tiny')
    for index, row in adata_X.obs.iterrows():
        dwg.add(dwg.circle(center=(row["X"], row["Y"]), r=5, fill='red'))
    dwg.save()
