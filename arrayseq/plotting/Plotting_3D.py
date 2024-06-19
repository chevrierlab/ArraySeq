
import plotly.graph_objs as go
import matplotlib.cm as cm
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
import plotly.io as pio
import random

def generate_custom_cmap(num_colors):
    """ Generate a custom colormap with `num_colors` distinct colors. """
    colors = cm.nipy_spectral(np.linspace(0, 1, num_colors))
    return [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c in colors]

def convert_matplotlib_to_plotly_cmap(cmap_name, num_colors):
    """ Convert a matplotlib colormap to a plotly colorscale """
    cmap = cm.get_cmap(cmap_name, num_colors)
    colors = [to_rgb(cmap(i)) for i in range(cmap.N)]
    plotly_colorscale = [[i/(len(colors)-1), f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'] for i, c in enumerate(colors)]
    return plotly_colorscale

def generate_random_cmap(num_colors=8):
    """
    Generate a user-defined colormap with `num_colors` distinct random colors.
    Returns a dictionary where keys are category labels as strings and values are color strings.
    """
    cmap = {}
    for i in range(1, num_colors + 1):
        color = [random.randint(0, 255) for _ in range(3)]
        cmap[str(i)] = f'rgb({color[0]}, {color[1]}, {color[2]})'
    return cmap

def plot_meta_3d(adata, meta_key, point_size=2, cmap=None, user_cmap=None, z_scale=0.5,
                 tick_width=2, tick_font_size=10, axis_labels=None, axis_label_font_size=12,
                 title=None, title_font_size=14, legend_font_size=15, camera_eye=dict(x=1.25, y=1.25, z=1.25),
                 save_path=None):
    """
    Generate a 3D scatter plot to display metadata from an AnnData object.

    This function creates a 3D scatter plot using the specified metadata from an AnnData object.
    The function allows customization of plot aspects such as point size, colormap, axis labels, and legend.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    meta_key : str
        The key in `adata.obs` to use for coloring the points.
    point_size : int, optional
        Size of the points in the scatter plot, by default 2.
    cmap : str, optional
        Colormap to use for continuous metadata, by default None. If `None`, 'Viridis' colormap is used.
    user_cmap : dict, optional
        Custom colormap for categorical metadata. The dictionary should map category names (as strings) to color values.
    z_scale : float, optional
        Scale for the z-axis, by default 0.5.
    tick_width : int, optional
        Width of the ticks on the axes, by default 2.
    tick_font_size : int, optional
        Font size of the tick labels, by default 10.
    axis_labels : list of str, optional
        Labels for the x, y, and z axes, by default ['X', 'Y', 'Z'].
    axis_label_font_size : int, optional
        Font size for the axis labels, by default 12.
    title : str, optional
        Title of the plot, by default None.
    title_font_size : int, optional
        Font size of the plot title, by default 14.
    legend_font_size : int, optional
        Font size of the legend text, by default 15.
    camera_eye : dict, optional
        Initial position of the camera, by default dict(x=1.25, y=1.25, z=1.25).
    save_path : str, optional
        Path to save the plot. Supported formats are '.html', '.png', '.jpg', '.jpeg', and '.svg'. By default, the plot is not saved.

    Returns
    -------
    None
        The function displays the plot and optionally saves it to a file.

    Notes
    -----
    - When `user_cmap` is provided, the function uses the specified colors for categorical metadata.
    - For continuous metadata, the function uses the specified colormap (`cmap`).
    - The function automatically adjusts the aspect ratio based on the range of x, y, and z coordinates.
    - The `save_path` parameter supports saving the plot in various formats including HTML and image files.

    Example
    -------
    >>> adata.obs["Z_Chat"] = adata.obs["Z"].astype(str)
    >>> user_cmap = {"1": "red", "2": "blue", "3": "green"}
    >>> plot_meta_3d(adata, meta_key='Z_Chat', user_cmap=user_cmap)
    """

    # Extract coordinates and metadata
    x_coords = adata.obs['X']
    y_coords = adata.obs['Y']
    z_coords = adata.obs['Z']
    metadata = adata.obs[meta_key]

    # Calculate the ranges for x, y, and z to set the aspect ratio
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Add a buffer around the data points
    buffer = 0.1
    x_buffer = x_range * buffer
    y_buffer = y_range * buffer
    z_buffer = z_range * buffer

    fig = go.Figure()

    # Use user-defined color map if provided
    if user_cmap:
        categories = pd.Categorical(metadata).categories
        colors = [user_cmap[str(cat)] for cat in categories]
        for i, cat in enumerate(categories):
            cat_data = adata.obs[adata.obs[meta_key] == cat]
            fig.add_trace(go.Scatter3d(
                x=cat_data['X'],
                y=cat_data['Y'],
                z=cat_data['Z'],
                mode='markers',
                name=str(cat),
                marker=dict(
                    size=point_size,
                    color=colors[i],
                    opacity=0.8
                ),
                showlegend=False
            ))
    else:
        if pd.api.types.is_numeric_dtype(metadata):
            color_data = metadata
            categories = []
            if cmap is None:
                cmap = 'Viridis'
            colorscale = convert_matplotlib_to_plotly_cmap(cmap, 10)
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color_data,
                    colorscale=colorscale,
                    colorbar=dict(
                        title=meta_key,
                        titleside='right',
                        titlefont=dict(size=legend_font_size),
                        tickfont=dict(size=legend_font_size),
                        thickness=15
                    ),
                    opacity=0.8
                ),
                showlegend=False
            ))
        else:
            categories = pd.Categorical(metadata).categories
            colorscale = convert_matplotlib_to_plotly_cmap(cmap, len(categories))
            color_data = pd.Categorical(metadata).codes
            colors = [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c in cm.get_cmap(cmap)(np.linspace(0, 1, len(categories)))]
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=color_data,
                    colorscale=colorscale,
                    opacity=0.8
                ),
                showlegend=False
            ))

    # Set default axis labels if not provided
    if axis_labels is None:
        axis_labels = ['X', 'Y', 'Z']

    # Update the layout to set the aspect ratio and customize the appearance
    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=x_range/max(x_range, y_range), y=y_range/max(x_range, y_range), z=z_scale),
            xaxis=dict(
                title=dict(text=axis_labels[0], font=dict(size=axis_label_font_size)),
                backgroundcolor='white',
                gridcolor='black',
                showbackground=True,
                range=[x_min - x_buffer, x_max + x_buffer],
                tickwidth=tick_width,
                tickfont=dict(size=tick_font_size)
            ),
            yaxis=dict(
                title=dict(text=axis_labels[1], font=dict(size=axis_label_font_size)),
                backgroundcolor='white',
                gridcolor='black',
                showbackground=True,
                range=[y_min - y_buffer, y_max + y_buffer],
                tickwidth=tick_width,
                tickfont=dict(size=tick_font_size)
            ),
            zaxis=dict(
                title=dict(text=axis_labels[2], font=dict(size=axis_label_font_size)),
                backgroundcolor='white',
                gridcolor='black',
                showbackground=True,
                range=[z_min - z_buffer, z_max + z_buffer * z_scale],
                tickwidth=tick_width,
                tickfont=dict(size=tick_font_size)
            ),
            camera=dict(
                eye=camera_eye
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # Remove margins
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=title_font_size)
        ),
        showlegend=True
    )

    # Add custom legend for categorical metadata
    if user_cmap or not pd.api.types.is_numeric_dtype(metadata):
        if not user_cmap:
            user_cmap = {str(cat): colors[i] for i, cat in enumerate(categories)}
        legend_items = []
        for cat in categories:
            legend_items.append(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color=user_cmap[str(cat)]),
                showlegend=True,
                name=str(cat)
            ))
        fig.add_traces(legend_items)
        # Update legend title
        fig.update_layout(legend_title=dict(text=meta_key, font=dict(size=legend_font_size)))

    # Save the plot if a save path is provided
    if save_path:
        if save_path.endswith('.html'):
            pio.write_html(fig, file=save_path, auto_open=False)
        elif save_path.endswith(('.png', '.jpg', '.jpeg', '.svg')):
            pio.write_image(fig, file=save_path)

    # Show the plot
    fig.show()




def plot_gene_3d(adata, gene, point_size=2, cmap='viridis', cmap_dynamic_alpha_color=None, 
                 z_scale=0.5, tick_width=2, tick_font_size=10, axis_labels=None, axis_label_font_size=12,
                 title=None, title_font_size=14, legend_font_size=15, camera_eye=dict(x=1.25, y=1.25, z=1.25),
                 save_path=None, **kwargs):
    """
    Generate a 3D scatter plot to display gene expression from an AnnData object with customizable legend and color mapping options.

    This function creates a 3D scatter plot using the specified gene expression data from an AnnData object.
    The function allows customization of plot aspects such as point size, colormap, axis labels, and legend.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    gene : str
        The gene name to visualize from the AnnData object.
    point_size : int, optional
        Size of the points in the scatter plot, by default 2.
    cmap : str, optional
        Colormap to use for continuous gene expression data, by default 'viridis'.
    cmap_dynamic_alpha_color : str, optional
        Hex color code for dynamic alpha color mapping, by default None.
    z_scale : float, optional
        Scale for the z-axis, by default 0.5.
    tick_width : int, optional
        Width of the ticks on the axes, by default 2.
    tick_font_size : int, optional
        Font size of the tick labels, by default 10.
    axis_labels : list of str, optional
        Labels for the x, y, and z axes, by default ['X', 'Y', 'Z'].
    axis_label_font_size : int, optional
        Font size for the axis labels, by default 12.
    title : str, optional
        Title of the plot, by default None.
    title_font_size : int, optional
        Font size of the plot title, by default 14.
    legend_font_size : int, optional
        Font size of the legend text, by default 15.
    camera_eye : dict, optional
        Initial position of the camera, by default dict(x=1.25, y=1.25, z=1.25).
    save_path : str, optional
        Path to save the plot. Supported formats are '.html', '.png', '.jpg', '.jpeg', and '.svg'. By default, the plot is not saved.
    **kwargs
        Additional keyword arguments for customization.

    Returns
    -------
    None
        The function displays the plot and optionally saves it to a file.

    Notes
    -----
    - The `cmap_dynamic_alpha_color` parameter allows for dynamic alpha blending based on gene expression levels. If provided, 
      this parameter overrides the standard colormap and uses a single color with varying transparency.
    - For continuous gene expression data, the function uses the specified colormap (`cmap`).
    - The function automatically adjusts the aspect ratio based on the range of x, y, and z coordinates.
    - The `save_path` parameter supports saving the plot in various formats including HTML and image files.

    Example
    -------
    >>> plot_gene_3d(adata, gene='Kap', cmap='plasma', point_size=3, title='GeneA Expression in 3D')
    """

    # Extract coordinates and gene expression data
    x_coords = adata.obs['X']
    y_coords = adata.obs['Y']
    z_coords = adata.obs['Z']
    gene_index = adata.var.index.get_loc(gene)
    gene_expression = adata.X[:, gene_index].toarray().flatten()
    max_expression = gene_expression.max()
    norm_expression = gene_expression / max_expression  # Normalize gene expression

    # Calculate the ranges for x, y, and z to set the aspect ratio
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Add a buffer around the data points
    buffer = 0.1
    x_buffer = x_range * buffer
    y_buffer = y_range * buffer
    z_buffer = z_range * buffer

    fig = go.Figure()

    # Check for dynamic alpha color mapping
    if cmap_dynamic_alpha_color:
        rgba_color = to_rgba(cmap_dynamic_alpha_color)  # Convert hex to RGBA
        colors = [(rgba_color[0], rgba_color[1], rgba_color[2], alpha) for alpha in norm_expression]
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=point_size,
                color=[f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})' for rgba in colors],
                opacity=1.0  # Opacity is handled per point
            ),
            showlegend=False
        ))

        # Create a colormap for alpha blending from white to the chosen color
        cmap_custom = LinearSegmentedColormap.from_list('custom_alpha', [(1, 1, 1, 0), rgba_color])
        sm = ScalarMappable(cmap=cmap_custom, norm=plt.Normalize(0, 1))
        sm.set_array([])
        fig.update_layout(coloraxis=dict(
            colorscale=[[0, 'rgba(255, 255, 255, 0)'], [1, f'rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})']],
            colorbar=dict(
                title='Expression level',
                titleside='right',
                titlefont=dict(size=legend_font_size),
                tickfont=dict(size=legend_font_size),
                thickness=15
            )
        ))
    else:
        colorscale = convert_matplotlib_to_plotly_cmap(cmap, 10)
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=point_size,
                color=norm_expression,
                colorscale=colorscale,
                colorbar=dict(
                    title=f'Expression of {gene}',
                    titleside='right',
                    titlefont=dict(size=legend_font_size),
                    tickfont=dict(size=legend_font_size),
                    thickness=15
                ),
                opacity=0.8
            ),
            showlegend=False
        ))

    # Set default axis labels if not provided
    if axis_labels is None:
        axis_labels = ['X', 'Y', 'Z']

    # Update the layout to set the aspect ratio and customize the appearance
    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=x_range/max(x_range, y_range), y=y_range/max(x_range, y_range), z=z_scale),
            xaxis=dict(
                title=dict(text=axis_labels[0], font=dict(size=axis_label_font_size)),
                backgroundcolor='white',
                gridcolor='black',
                showbackground=True,
                range=[x_min - x_buffer, x_max + x_buffer],
                tickwidth=tick_width,
                tickfont=dict(size=tick_font_size)
            ),
            yaxis=dict(
                title=dict(text=axis_labels[1], font=dict(size=axis_label_font_size)),
                backgroundcolor='white',
                gridcolor='black',
                showbackground=True,
                range=[y_min - y_buffer, y_max + y_buffer],
                tickwidth=tick_width,
                tickfont=dict(size=tick_font_size)
            ),
            zaxis=dict(
                title=dict(text=axis_labels[2], font=dict(size=axis_label_font_size)),
                backgroundcolor='white',
                gridcolor='black',
                showbackground=True,
                range=[z_min - z_buffer, z_max + z_buffer * z_scale],
                tickwidth=tick_width,
                tickfont=dict(size=tick_font_size)
            ),
            camera=dict(
                eye=camera_eye
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # Remove margins
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=title_font_size)
        ),
        showlegend=False
    )

    # Save the plot if a save path is provided
    if save_path:
        if save_path.endswith('.html'):
            pio.write_html(fig, file=save_path, auto_open=False)
        elif save_path.endswith(('.png', '.jpg', '.jpeg', '.svg')):
            pio.write_image(fig, file=save_path)

    # Show the plot
    fig.show()


def plot_HE_3d(adata, point_size=2, z_scale=0.5, tick_width=2, tick_font_size=10, axis_labels=None,
               axis_label_font_size=12, title=None, title_font_size=14, legend_font_size=15,
               camera_eye=dict(x=1.25, y=1.25, z=1.25), save_path=None, **kwargs):
    """
    Generate a 3D scatter plot to display H&E images using RGB values from image data and coordinates from an AnnData object.

    This function creates a 3D scatter plot where each point is colored according to the RGB values extracted from H&E images.
    The function utilizes the spatial coordinates (X, Y, Z) from the AnnData object to position each point in the 3D space.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial coordinates and image data.
    point_size : int, optional
        Size of the points in the scatter plot, by default 2.
    z_scale : float, optional
        Scale for the z-axis, by default 0.5.
    tick_width : int, optional
        Width of the ticks on the axes, by default 2.
    tick_font_size : int, optional
        Font size of the tick labels, by default 10.
    axis_labels : list of str, optional
        Labels for the x, y, and z axes, by default ['X', 'Y', 'Z'].
    axis_label_font_size : int, optional
        Font size for the axis labels, by default 12.
    title : str, optional
        Title of the plot, by default None.
    title_font_size : int, optional
        Font size of the plot title, by default 14.
    legend_font_size : int, optional
        Font size of the legend text, by default 15.
    camera_eye : dict, optional
        Initial position of the camera, by default dict(x=1.25, y=1.25, z=1.25).
    save_path : str, optional
        Path to save the plot. Supported formats are '.html', '.png', '.jpg', '.jpeg', and '.svg'. By default, the plot is not saved.
    **kwargs
        Additional keyword arguments for customization.

    Returns
    -------
    None
        The function displays the plot and optionally saves it to a file.

    Notes
    -----
    - The RGB values for each point are extracted from the H&E images stored in `adata.uns["Image"]`.
    - The `adata.obs` DataFrame should contain columns 'X', 'Y', and 'Z' for the spatial coordinates.
    - The `adata.uns["Image"]` should be a list of numpy arrays representing the H&E images, where each array corresponds to a unique Z level.
    - The function ensures that the axis ranges and aspect ratios are set appropriately to visualize the 3D spatial relationships.

    Example
    -------
    >>> plot_HE_3d(adata, point_size=3, title='3D H&E Image Plot')
    """
    # Extract coordinates and metadata
    x_coords = adata.obs['X'].round().astype(int)
    y_coords = adata.obs['Y'].round().astype(int)
    z_coords = adata.obs['Z']
    
    # Get unique Z values and sort them
    unique_z_values = sorted(adata.obs['Z'].unique())

    # Initialize a list to hold the RGB values for each point
    rgb_colors = []

    # Extract RGB values for each (x, y) from the corresponding image
    for z in unique_z_values:
        z_index = int(z) - 1  # Adjust for 0-based indexing
        image = adata.uns["Image"][z_index]
        
        z_mask = adata.obs['Z'] == z
        x_z = x_coords[z_mask]
        y_z = y_coords[z_mask]
        
        for x, y in zip(x_z, y_z):
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:  # Ensure coordinates are within bounds
                rgb = image[y, x, :3]  # Extract RGB values
                rgb_colors.append(f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})')
            else:
                rgb_colors.append('rgb(0, 0, 0)')  # Default to black if out of bounds

    # Calculate the ranges for x, y, and z to set the aspect ratio
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # Add a buffer around the data points
    buffer = 0.1
    x_buffer = x_range * buffer
    y_buffer = y_range * buffer
    z_buffer = z_range * buffer

    # Create the 3D scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=point_size,
            color=rgb_colors,
            opacity=1.0
        ),
        showlegend=False
    ))

    # Set default axis labels if not provided
    if axis_labels is None:
        axis_labels = ['X', 'Y', 'Z']

    # Update the layout to set the aspect ratio and customize the appearance
    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=x_range/max(x_range, y_range), y=y_range/max(x_range, y_range), z=z_scale),
            xaxis=dict(
                title=dict(text=axis_labels[0], font=dict(size=axis_label_font_size)),
                backgroundcolor='white',
                gridcolor='black',
                showbackground=True,
                range=[x_min - x_buffer, x_max + x_buffer],
                tickwidth=tick_width,
                tickfont=dict(size=tick_font_size)
            ),
            yaxis=dict(
                title=dict(text=axis_labels[1], font=dict(size=axis_label_font_size)),
                backgroundcolor='white',
                gridcolor='black',
                showbackground=True,
                range=[y_min - y_buffer, y_max + y_buffer],
                tickwidth=tick_width,
                tickfont=dict(size=tick_font_size)
            ),
            zaxis=dict(
                title=dict(text=axis_labels[2], font=dict(size=axis_label_font_size)),
                backgroundcolor='white',
                gridcolor='black',
                showbackground=True,
                range=[z_min - z_buffer, z_max + z_buffer * z_scale],
                tickwidth=tick_width,
                tickfont=dict(size=tick_font_size)
            ),
            camera=dict(
                eye=camera_eye
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),  # Remove margins
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=title_font_size)
        ),
        showlegend=False
    )

    # Save the plot if a save path is provided
    if save_path:
        if save_path.endswith('.html'):
            pio.write_html(fig, file=save_path, auto_open=False)
        elif save_path.endswith(('.png', '.jpg', '.jpeg', '.svg')):
            pio.write_image(fig, file=save_path)

    # Show the plot
    fig.show()
    
