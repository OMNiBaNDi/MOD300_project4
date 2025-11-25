import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap
from numpy import numpy as np


# Task 2 - Reusable Milky way sector plot and fig function
def make_mw_sector(
    center,
    radius_x_arcsec,
    radius_y_arcsec=None,
    background="Mellinger color optical survey",
    projection="equirectangular",
    grayscale=False,
    figsize=(5, 5),
    savepath=None,
    show=True,
):
    """
    Create and optionally save a Milky Way sky map sector.

    Parameters
    ----------
    center : str or tuple
        Center of the map. Can be an object name like "M31" or a tuple
        of (RA, Dec) with astropy units.
    radius_x_arcsec : float
        Half-size of the field of view along the x axis, in arcseconds.
    radius_y_arcsec : float or None
        Half-size along the y axis in arcseconds. If None, uses same as x.
    background : str
        Background survey name understood by MWSkyMap.
    projection : str
        Sky projection (e.g. "gnomonic", "equirectangular").
    grayscale : bool
        If True, uses a grayscale rendering.
    figsize : tuple
        Figure size for matplotlib.
    savepath : str or None
        If provided, save the figure to this path.
    show : bool
        If True, call plt.show().

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if radius_y_arcsec is None:
        radius_y_arcsec = radius_x_arcsec

    radius = (radius_x_arcsec , radius_y_arcsec)* u.arcsec

    mw = MWSkyMap(
        grayscale=grayscale,
        projection=projection,
        background=background,
        center=center,
        radius=radius,
        figsize=figsize,
    )

    fig, ax = plt.subplots(figsize=figsize)
    mw.transform(ax)

    if savepath is not None:
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()

    return fig, ax

# Task 3
def plt2rgbarr(fig):
    """
    A function to transform a matplotlib to a 3d rgb np.array 

    Input
    -----
    fig: matplotlib.figure.Figure
        The plot that we want to encode.        

    Output
    ------
    np.array(ndim, ndim, 3): A 3d map of each pixel in a rgb encoding (the three dimensions are x, y, and rgb)
    
    """
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr[:, :, :3]


# Task 4

img_array = plt2rgbarr(fig)
print(img_array.shape)  

# A grey encoding
grey = np.sum(img_array[: , : , :] * np.array([0.299, 0.587, 0.114]), axis=2)  # From RGB to grey
x, y = [], []
for ig, g in enumerate(grey):
    for ij, j in enumerate(g):
        if j > 230:
            x.append(ig)
            y.append(ij)

plt.scatter(x, y, s=0.1)