from wand.image import Image
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import os
import svgwrite

"""
Adjust this code (in addition to other scripts) to assign parsers to variables and the input those variables into the functions
This will let you import the functions into other scripts and use them there
"""


#Takes User Input
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Generate SVG of UMI/spot from raw STARsolo data')
    
    parser.add_argument("-s", "--sample_name", help="Name of sample")

    parser.add_argument("-i", "--input", required=True,
                        help="Path to gzipped files from STARsolo")

    parser.add_argument("-c", "--coords", required=True,
                        help="Path to ST Coordinates file")

    parser.add_argument("--dpi", type=int, default=225,
                        help="Resolution of generated png image")               

    parser.add_argument("--image_output", default=os.path.join(os.getcwd(), 'spot_images'), 
                        help='Path to svg and png output folder')

    parser.add_argument('--counts_output', default=os.path.join(os.getcwd()), 
                        help='Path to scanpy output files')                    

    args = parser.parse_args()



#Read in STARsolo output and write svg file and scanpy object
def write_svg():
    adata = sc.read_10x_mtx(args.input, var_names='gene_symbols', cache=True) 
    adata.obs["UMI"] = np.concatenate(adata.X.sum(axis=1).tolist()).flat

    #Read in coordinates and join to the scanpy object by barcode
    coords = pd.read_csv(args.coords)
    adata.obs = adata.obs.join(coords.set_index("barcodes"))

    #Scale data to allow for SVG creation
    x = adata.obs["x_image"].multiply(100)
    y = adata.obs["y_image"].multiply(100)

    #Scale color intensity of each spot based on UMI/Spot
    color = adata.obs["UMI"].divide(adata.obs["UMI"].max()).multiply(255)

    #Write SVG by making a circle graphic for each spot on the array colored by UMI/Spot
    dwg = svgwrite.Drawing("{image_output}_{sample_name}.svg".format(image_output = args.image_output, sample_name = args.sample_name))
    for i in range(0,len(adata.obs)):
        dwg.add(dwg.circle(center=(x[i], y[i]), r=1, stroke=svgwrite.rgb(round(color[i]), 0, 0, '%'), fill=svgwrite.rgb(round(color[i]), 0, 0, '%')))
    dwg.save()

    #Save Scanpy object:
    adata.write("{counts_output}{sample_name}.h5ad".format(counts_output = args.counts_output, sample_name = args.sample_name))


#Read in SVG object and convert to PNG
def write_png():
    with Image(filename="{image_output}_{sample_name}.svg".format(image_output = args.image_output, sample_name = args.sample_name), 
    background="transparent", resolution=args.dpi) as image:
        image.save(filename='{image_output}_{sample_name}.png'.format(image_output = args.image_output, sample_name = args.sample_name))


write_svg()
write_png()
