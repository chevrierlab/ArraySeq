import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from skimage import exposure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import to_rgba


def generate_custom_cmap(num_colors):
    """ Generate a custom colormap with `num_colors` distinct colors. """
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_colors))
    return LinearSegmentedColormap.from_list("custom_cmap", colors, num_colors)

def plot_meta(adata, 
              meta_key, 
              image_key='Image', 
              point_size=50, 
              point_alpha=1, 
              cmap=None, 
              bw=False, 
              image_alpha=0.5, 
              plot_height=6,
              show_ticks=True, 
              show_labels=True, 
              show_title=True, 
              border_width=2,
              title_text=None,
              label_font_size=12,
              title_font_size=14,
              legend_font_size=12,
              font_weight='normal', 
              axis_tick_size=10,
              colorbar_tick_size=10, 
              axis_tick_width=2, 
              colorbar_tick_width=2,
              colorbar_shrink=0.75, 
              colorbar_aspect=20, 
              colorbar_pad=0.01, 
              legend_point_size=10,
              user_cmap=None):
     """
    Enhanced plot function to display metadata over an image with options for customization, including user-defined color maps.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing the spatial coordinates and metadata.
    meta_key : str
        Key for the metadata in `adata.obs` to be plotted.
    image_key : str, optional, default='Image'
        Key for the image data in `adata.uns`.
    point_size : int, optional, default=50
        Size of the points in the scatter plot.
    point_alpha : float, optional, default=1
        Alpha (opacity) of the points in the scatter plot.
    cmap : str or Colormap, optional
        Colormap to use for the plot. If `user_cmap` is provided, this is ignored.
    bw : bool, optional, default=False
        If True, convert the image to black and white.
    image_alpha : float, optional, default=0.5
        Alpha (opacity) of the background image.
    plot_height : int, optional, default=6
        Height of the plot in inches.
    show_ticks : bool, optional, default=True
        If True, display the axis ticks.
    show_labels : bool, optional, default=True
        If True, display the axis labels.
    show_title : bool, optional, default=True
        If True, display the plot title.
    border_width : int, optional, default=2
        Width of the plot border.
    title_text : str, optional
        Custom title text for the plot. If None, a default title is used.
    label_font_size : int, optional, default=12
        Font size for the axis labels.
    title_font_size : int, optional, default=14
        Font size for the plot title.
    legend_font_size : int, optional, default=12
        Font size for the legend text.
    font_weight : str, optional, default='normal'
        Font weight for the labels and title.
    axis_tick_size : int, optional, default=10
        Font size for the axis ticks.
    colorbar_tick_size : int, optional, default=10
        Font size for the colorbar ticks.
    axis_tick_width : int, optional, default=2
        Width of the axis ticks.
    colorbar_tick_width : int, optional, default=2
        Width of the colorbar ticks.
    colorbar_shrink : float, optional, default=0.75
        Shrink factor for the colorbar.
    colorbar_aspect : int, optional, default=20
        Aspect ratio for the colorbar.
    colorbar_pad : float, optional, default=0.01
        Padding between the plot and the colorbar.
    legend_point_size : int, optional, default=10
        Size of the points in the legend.
    user_cmap : dict, optional
        User-defined colormap for categorical data. Keys should be the metadata categories and values should be the corresponding colors.

    Returns
    -------
    None
        Displays the plot.

    Raises
    ------
    ValueError
        If `user_cmap` is provided but not properly formatted.

    Notes
    -----
    - The function supports both numeric and categorical metadata.
    - If `user_cmap` is provided, it takes precedence over the `cmap` parameter.
    - If `bw` is True, the background image is converted to black and white.
    - The function allows extensive customization of plot appearance through various parameters.

    Example
    -------
    >>> plot_meta(adata=my_adata, meta_key='cell_type')
    """
    # Load and process the image
    image = adata.uns[image_key]
    if bw:
        image = np.mean(image, axis=2)
        image = image / 255.0  # Normalize to [0, 1] if originally in [0, 255]
    image = exposure.adjust_gamma(image, image_alpha)  # Adjust gamma for lightening

    # Extract coordinates and metadata
    x_coords = adata.obs['X']
    y_coords = adata.obs['Y']
    metadata = adata.obs[meta_key]

    # Use user-defined color map if provided
    if user_cmap:
        categories = pd.Categorical(metadata).categories
        color_data = [user_cmap[str(cat)] for cat in metadata]
        cmap = LinearSegmentedColormap.from_list("user_defined_cmap", list(user_cmap.values()), N=len(user_cmap))
        norm = None  # Color data directly provides colors
    else:
        # Handle non-numeric metadata; factorize if not already categorical
        if pd.api.types.is_numeric_dtype(metadata):
            color_data = metadata
            norm = Normalize(vmin=color_data.min(), vmax=color_data.max())
            categories = []  # Numeric data doesn't use categories
            if cmap is None:
                cmap = 'viridis'
        else:
            categories = pd.Categorical(metadata).categories
            color_data = pd.Categorical(metadata).codes
            if cmap is None:
                cmap = 'tab10'
            cmap_base = plt.get_cmap(cmap)
            if len(categories) > cmap_base.N:
                print(f"Warning: The number of categories ({len(categories)}) exceeds the number of colors in the selected colormap ({cmap_base.N}). Generating a custom colormap.")
                cmap = generate_custom_cmap(len(categories))
            else:
                cmap = plt.cm.get_cmap(cmap, len(categories))
            norm = Normalize(vmin=0, vmax=len(categories) - 1)

    # Setup the plot
    fig, ax = plt.subplots(figsize=(plot_height * 1.5, plot_height))
    ax.imshow(image, cmap='gray' if bw else None, aspect='equal')
    scatter = ax.scatter(x_coords, y_coords, c=color_data, s=point_size, cmap=cmap, alpha=point_alpha, norm=norm if norm else matplotlib.colors.NoNorm())

    # Create a legend for categorical data
    if not user_cmap and categories:
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=legend_point_size, label=str(cat)) for i, cat in enumerate(categories)]
        legend = ax.legend(handles=handles, title=meta_key, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legend_font_size)
        legend.get_title().set_fontsize(legend_font_size)

    # Set plot labels and title
    if show_labels:
        ax.set_xlabel('X Coordinate', fontsize=label_font_size, weight=font_weight)
        ax.set_ylabel('Y Coordinate', fontsize=label_font_size, weight=font_weight)
    if show_title:
        ax.set_title(title_text if title_text else f'Spatial Plot of {meta_key}', fontsize=title_font_size, weight=font_weight)

    # Hide or show ticks
    if show_ticks:
        ax.tick_params(axis='both', which='major', labelsize=axis_tick_size, width=axis_tick_width)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Optional border
    for _, spine in ax.spines.items():
        spine.set_linewidth(border_width)
        spine.set_color('black')

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Add colorbar if needed
    if not user_cmap and pd.api.types.is_numeric_dtype(metadata):
        cbar = plt.colorbar(scatter, ax=ax, shrink=colorbar_shrink, aspect=colorbar_aspect, pad=colorbar_pad)
        cbar.ax.tick_params(labelsize=colorbar_tick_size, width=colorbar_tick_width)
        cbar.set_label(meta_key, fontsize=legend_font_size, weight=font_weight)

    # Show the plot
    plt.show()
    


def plot_gene(adata, 
              gene,
              image_key='Image', 
              point_size=5, 
              cmap='viridis', 
              cmap_dynamic_alpha_color=None,
              bw=False, 
              image_alpha=0.5, 
              plot_height=6,
              show_ticks=True, 
              show_labels=True, 
              show_title=True, 
              border_width=2,
              title_text=None,
              label_font_size=12,
              title_font_size=14,
              font_weight='normal',
              legend_font_size=12,
              colorbar_shrink=0.75, 
              colorbar_aspect=20, 
              colorbar_pad=0.01):
    """
    Function to display gene expression over an image from an AnnData object with customizable legend and color mapping options.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing the spatial coordinates and gene expression data.
    gene : str
        Name of the gene to be plotted.
    image_key : str, optional, default='Image'
        Key for the image data in `adata.uns`.
    point_size : int, optional, default=5
        Size of the points in the scatter plot.
    cmap : str or Colormap, optional
        Colormap to use for the plot. Ignored if `cmap_dynamic_alpha_color` is provided.
    cmap_dynamic_alpha_color : str or tuple, optional
        Color to use for dynamic alpha blending, specified as a hex string or an RGBA tuple.
    bw : bool, optional, default=False
        If True, convert the image to black and white.
    image_alpha : float, optional, default=0.5
        Alpha (opacity) of the background image.
    plot_height : int, optional, default=6
        Height of the plot in inches.
    show_ticks : bool, optional, default=True
        If True, display the axis ticks.
    show_labels : bool, optional, default=True
        If True, display the axis labels.
    show_title : bool, optional, default=True
        If True, display the plot title.
    border_width : int, optional, default=2
        Width of the plot border.
    title_text : str, optional
        Custom title text for the plot. If None, a default title is used.
    label_font_size : int, optional, default=12
        Font size for the axis labels.
    title_font_size : int, optional, default=14
        Font size for the plot title.
    font_weight : str, optional, default='normal'
        Font weight for the labels and title.
    legend_font_size : int, optional, default=12
        Font size for the legend text.
    colorbar_shrink : float, optional, default=0.75
        Shrink factor for the colorbar.
    colorbar_aspect : int, optional, default=20
        Aspect ratio for the colorbar.
    colorbar_pad : float, optional, default=0.01
        Padding between the plot and the colorbar.

    Returns
    -------
    None
        Displays the plot.

    Raises
    ------
    ValueError
        If the gene name is not found in `adata.var`.

    Notes
    -----
    - The function supports both standard color mapping and dynamic alpha blending.
    - If `bw` is True, the background image is converted to black and white.
    - The function allows extensive customization of plot appearance through various parameters.

    Example
    -------
    >>> plot_gene(adata=my_adata, gene='Kap')
    """
    # Load and process the image
    image = adata.uns[image_key]
    if bw:
        image = np.mean(image, axis=2)
        image = image / 255.0  # Normalize to [0, 1] if originally in [0, 255]

    image = exposure.adjust_gamma(image, image_alpha)  # Adjust gamma for lightening

    # Extract coordinates and gene expression data
    x_coords = adata.obs['X']
    y_coords = adata.obs['Y']
    gene_index = adata.var.index.get_loc(gene)
    gene_expression = adata.X[:, gene_index].toarray().flatten()
    max_expression = gene_expression.max()
    norm_expression = gene_expression / max_expression  # Normalize gene expression

    # Setup the plot
    fig, ax = plt.subplots(figsize=(plot_height * 1.5, plot_height))
    ax.imshow(image, cmap='gray' if bw else None, aspect='equal')

    # Check for dynamic alpha color mapping
    if cmap_dynamic_alpha_color:
        rgba_color = to_rgba(cmap_dynamic_alpha_color)  # Convert hex to RGBA
        colors = [(rgba_color[0], rgba_color[1], rgba_color[2], alpha) for alpha in norm_expression]
        scatter = ax.scatter(x_coords, y_coords, color=colors, s=point_size, edgecolor='none')

        # Create a colormap for alpha blending from white to the chosen color
        cmap_custom = LinearSegmentedColormap.from_list('custom_alpha', [(1, 1, 1, 0), rgba_color])
        sm = ScalarMappable(cmap=cmap_custom, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=colorbar_shrink, aspect=colorbar_aspect, pad=colorbar_pad)
        cbar.set_label('Expression level', fontsize=legend_font_size)
    else:
        scatter = ax.scatter(x_coords, y_coords, c=norm_expression, s=point_size, cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        # Standard colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=colorbar_shrink, aspect=colorbar_aspect, pad=colorbar_pad)
        cbar.set_label(f'Expression of {gene}', fontsize=legend_font_size)

    # Set plot labels and title
    if show_labels:
        ax.set_xlabel('X Coordinate', fontsize=label_font_size, weight=font_weight)
        ax.set_ylabel('Y Coordinate', fontsize=label_font_size, weight=font_weight)
    if show_title:
        ax.set_title(title_text if title_text else f'Spatial Plot of {gene}', fontsize=title_font_size, weight=font_weight)

    # Manage ticks
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    # Optional border
    for _, spine in ax.spines.items():
        spine.set_linewidth(border_width)
        spine.set_color('black')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Show the plot
    plt.show()

    


def plot_multi_gene(adata, genes, image_key='Image', point_size=5, bw=False, image_alpha=0.5, plot_height=6,
                    show_ticks=True, show_labels=True, show_title=True, border_width=2,
                    title_text=None, label_font_size=12, title_font_size=14, font_weight='normal',
                    colorbar_spacing=0.1, legend_font_size=12, legend_title_font_size=12, 
                    colorbar_tick_font_size=10, colorbar_pad=0.1):
   """
    Function to display gene expression over an image from an AnnData object with RGB color mapping for 2 or 3 genes,
    including customizable legend and color bar features.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing the spatial coordinates and gene expression data.
    genes : list of str
        List of 2 or 3 gene names to be plotted with RGB color mapping.
    image_key : str, optional, default='Image'
        Key for the image data in `adata.uns`.
    point_size : int, optional, default=5
        Size of the points in the scatter plot.
    bw : bool, optional, default=False
        If True, convert the image to black and white.
    image_alpha : float, optional, default=0.5
        Alpha (opacity) of the background image.
    plot_height : int, optional, default=6
        Height of the plot in inches.
    show_ticks : bool, optional, default=True
        If True, display the axis ticks.
    show_labels : bool, optional, default=True
        If True, display the axis labels.
    show_title : bool, optional, default=True
        If True, display the plot title.
    border_width : int, optional, default=2
        Width of the plot border.
    title_text : str, optional
        Custom title text for the plot. If None, a default title is used.
    label_font_size : int, optional, default=12
        Font size for the axis labels.
    title_font_size : int, optional, default=14
        Font size for the plot title.
    font_weight : str, optional, default='normal'
        Font weight for the labels and title.
    colorbar_spacing : float, optional, default=0.1
        Spacing between the color bars for each gene.
    legend_font_size : int, optional, default=12
        Font size for the legend text.
    legend_title_font_size : int, optional, default=12
        Font size for the legend title text.
    colorbar_tick_font_size : int, optional, default=10
        Font size for the color bar tick labels.
    colorbar_pad : float, optional, default=0.1
        Padding between the plot and the color bar.

    Returns
    -------
    None
        Displays the plot.

    Raises
    ------
    ValueError
        If the number of genes provided is not 2 or 3.

    Notes
    -----
    - The function supports RGB color mapping for visualizing the expression of 2 or 3 genes.
    - If `bw` is True, the background image is converted to black and white.
    - The function allows extensive customization of plot appearance through various parameters.

    Example
    -------
    >>> plot_multi_gene(adata, genes=['Napsa', 'Kap', 'Slc12a1'])
    """
    if not (2 <= len(genes) <= 3):
        raise ValueError("Please provide 2 or 3 genes for RGB color mapping.")
    
    # Load and process the image
    image = adata.uns[image_key]
    if bw:
        image = np.mean(image, axis=2)  # Convert to grayscale by averaging channels
    image = exposure.adjust_gamma(image, image_alpha)  # Adjust gamma for lightening

    # Extract coordinates
    x_coords = adata.obs['X']
    y_coords = adata.obs['Y']

    # Normalize and scale gene expression data for each gene
    rgb_values = []
    colorbars = []
    for gene in genes:
        gene_index = adata.var.index.get_loc(gene)
        expression = adata.X[:, gene_index].toarray().flatten()
        max_expression = np.percentile(expression, 99)
        norm_expression = expression / max_expression
        norm_expression[norm_expression > 1] = 1
        rgb_values.append(norm_expression)
        colorbars.append((norm_expression, max_expression))

    rgb_values = np.vstack(rgb_values).T
    if len(genes) == 2:
        rgb_values = np.column_stack([rgb_values, np.zeros_like(rgb_values[:, 0])])  # Add zeros for blue

    # Setup the plot
    fig, ax = plt.subplots(figsize=(plot_height * 1.5, plot_height))
    ax.imshow(image, cmap='gray' if bw else None, aspect='equal')
    scatter = ax.scatter(x_coords, y_coords, color=rgb_values, s=point_size, edgecolor='none')

    # Calculate the vertical centering for color bars
    total_colorbar_height = len(genes) * 0.03 + (len(genes) - 1) * colorbar_spacing
    start_pos = 0.5 - total_colorbar_height / 2  # Centered position

    # Add color bars for each gene
    for idx, (expression, max_exp) in enumerate(colorbars):
        color = ['red', 'green', 'blue'][idx]
        cax = fig.add_axes([0.88, start_pos + idx * (0.03 + colorbar_spacing), 0.1, 0.03])
        cmap = LinearSegmentedColormap.from_list("gene_cmap", ['black', color])
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_exp))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal', pad=colorbar_pad)
        cbar.set_label(f'{genes[idx]} Expression', fontsize=legend_title_font_size)
        cbar.ax.tick_params(labelsize=colorbar_tick_font_size)  # Set font size for color bar ticks
        cbar.ax.xaxis.set_label_position('top')  # Set label position to top

    # Set plot labels and title
    if show_labels:
        ax.set_xlabel('X Coordinate', fontsize=label_font_size, weight=font_weight)
        ax.set_ylabel('Y Coordinate', fontsize=label_font_size, weight=font_weight)
    if show_title:
        ax.set_title(title_text if title_text else f'Spatial Plot of {" vs ".join(genes)}', fontsize=title_font_size, weight=font_weight)

    # Manage ticks
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    # Optional border
    for _, spine in ax.spines.items():
        spine.set_linewidth(border_width)
        spine.set_color('black')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Show the plot
    plt.show()

