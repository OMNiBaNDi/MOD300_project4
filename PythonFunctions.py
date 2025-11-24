import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap


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

    radius = (radius_x_arcsec * u.arcsec, radius_y_arcsec * u.arcsec)

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
