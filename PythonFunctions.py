import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap
import numpy as np
from sklearn.cluster import KMeans


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

# Converts an RGB image array to Greyscale array
def rgb_to_greyscale(img_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image array to a 2D greyscale array using standard
    luminance weights.

    Parameters
    ----------
    img_array : np.ndarray
        Array of shape (H, W, 3) with RGB values in uint8.

    Returns
    -------
    np.ndarray
        Array of shape (H, W) with greyscale intensity values.
    """
    #Using standard weights for greyscale
    weights = np.array([0.299, 0.587, 0.114], dtype = float)
    grey = np.tensordot(img_array[..., :3], weights, axes = ([-1], [0]))
    return grey

def encode_greyscale_brightness(img_array: np.ndarray, grey_threshold: float = 230.0):
    """
    Encode an RGB image by greyscale brightness, keeping only bright pixels.

    Parameters
    ----------
    img_array : np.ndarray
        RGB image array of shape (H, W, 3).
    grey_threshold : float
        Greyscale threshold. Only pixels with grey > threshold are kept.

    Returns
    -------
    features : np.ndarray
        Array of shape (N, 1) with greyscale intensities of bright pixels.
        This is the feature representation (encoding) for clustering.
    coords : np.ndarray
        Array of shape (N, 2) with (x, y) pixel coordinates for each feature.
        coords[i] = [x_i, y_i], where x is column index and y is row index.
    grey : np.ndarray
        The full greyscale image of shape (H, W) for inspection.
    """
    grey = rgb_to_greyscale(img_array)

    # storing indices of bright pixels
    ys, xs = np.where(grey > grey_threshold)

    # Feature: brightness of each bright pixel
    features = grey[ys, xs].reshape(-1, 1)

    # Store coordinates for plotting
    coords = np.stack([xs, ys], axis = 1)
    
    return features, coords, grey


# Task 5

# Kmeans cluster features
def kmeans_cluster_features(
    features: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
):
    """
    Cluster feature vectors using the K-Means algorithm.

    Parameters
    ----------
    features : np.ndarray
        Array of shape (N, n_features) containing the feature vectors
        to be clustered. In our case we often use brightness features
        of shape (N, 1).
    n_clusters : int
        Number of clusters (K) to find.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    labels : np.ndarray
        Cluster labels for each feature (shape (N,)).
    model : sklearn.cluster.KMeans
        The fitted KMeans model (contains cluster centers, inertia, etc.).
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = kmeans.fit_predict(features)
    return labels, kmeans

def run_kmeans_on_greyscale(
        img_array: np.ndarray,
        grey_threshold: float = 230.0,
        n_clusters: int = 3,
        random_state: int = 0,
):
    """
    Takes an RGB image array, encodes it to greyscale and uses K-means to cluster.

    Parameters:
        img_array: np.ndarray
            an RGB image array
        grey_threshold: float
            Greyscale threshold, only pixels with grey > threshold are used
        n_clusters: int
            Number of clusters K for K-means
        random_state: int
            Random seed for reproducibility

    Returns:
        labels: np.ndarray
            cluster label for each bright pixel
        coords: np.ndarray
            pixel coordinates for each bright pixel
        features: np.ndarray
            Brightness features used by K-means
    """
    # Encode into Greyscale and store in variables features, coords, grey
    features, coords, grey = encode_greyscale_brightness(img_array, grey_threshold=grey_threshold)

    # Run K-means clustering on Greyscale pixels and store into variables labels and kmeans_model
    labels, kmeans_model = kmeans_cluster_features(features, n_clusters=n_clusters, random_state=random_state)

    return labels, coords, features, grey, kmeans_model


# Task 6