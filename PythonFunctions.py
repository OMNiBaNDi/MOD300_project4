import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWSkyMap
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

CLUSTER_OUTLINE_COLORS = ["red", "lime", "cyan"]



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

    Parameters:
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

    Returns:
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

    Parameters:
    fig: matplotlib.figure.Figure
        The plot that we want to encode.        

    Returns:
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

    Parameters:
        img_array : np.ndarray
            Array of shape (H, W, 3) with RGB values in uint8.

    Returns:
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

    Parameters:
        img_array : np.ndarray
            RGB image array of shape (H, W, 3).
        grey_threshold : float
            Greyscale threshold. Only pixels with grey > threshold are kept.

    Returns:
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

    Parameters:
        features : np.ndarray
            Array of shape (N, n_features) containing the feature vectors
            to be clustered. In our case we often use brightness features
            of shape (N, 1).
        n_clusters : int
            Number of clusters (K) to find.
        random_state : int
            Random seed for reproducibility.

    Returns:
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


# Task 6 - Greyscale encoding

def overimpose_clusters_on_image(
        img_array: np.ndarray,
        coords: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        figsize: tuple[float, float] = (5, 5),
        point_size: float = 2.0,
        alpha: float = 0.6,
):
    """
    plot the original RGB image and over-impose the clustered pixels

    Parameters:
        img_array : np.ndarray
        RGB image array of shape (H, W, 3).
        coords : np.ndarray
            Pixel coordinates of bright pixels, shape (N, 2),
            where coords[i] = [x_i, y_i].
        labels : np.ndarray
            Cluster labels for each bright pixel, shape (N,).
        n_clusters : int
            Number of clusters (K) used in K-Means. Used mainly for legend.
        figsize : tuple
            Figure size for matplotlib.
        point_size : float
            Size of the scatter points.
        alpha : float
            Transparency for the scatter overlay.

    Returns:
        fig, ax : matplotlib Figure and Axes
            The created figure and axes.
    """

    fig, ax = plt.subplots(figsize = figsize)

    # Show original RGB image as the background
    ax.imshow(img_array, origin = "upper")

    # Map each label to an outline color
    edgecolor_array = [
        CLUSTER_OUTLINE_COLORS[int(l) % len(CLUSTER_OUTLINE_COLORS)]
        for l in labels
    ]

    # Over-imposing the clustered pixels
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        facecolors="white",
        edgecolors=edgecolor_array,
        s=point_size,
        alpha=alpha,
        linewidths=0.6,
    )

    ax.set_axis_off()
    ax.set_title(f"K-means Clusters (K = {n_clusters})", fontsize = 10)

    return fig, ax

# Task 7 - HSV color encoding

def encode_hsv_color(
    img_array: np.ndarray,
    grey_threshold: float = 230.0,
):
    """
    Encoding an RGB image by HSV color information. The pixels that are over the greyscale threshold
    are stored in (Hue, Saturation) feature vector.

    Parameters:
        img_array : np.ndarray
            RGB image array
        grey_treshold: float
            Greyscale threshold on a 0-255 scale. Only pixels with greyscale above threshold are kept.

    Returns:
        features : np.ndarray
            An array with [Hue, Saturation] for each selected pixel.
        coords : np.ndarray
            Array with pixel coordinates for each selected pixel.
        hsv_image : np.ndarry
            HSV image

    """
    # Finding and storing pixels above greyscale_threshold
    grey = rgb_to_greyscale(img_array)

    # Storing the indices of the bright pixels
    ys, xs = np.where(grey > grey_threshold)

    # Normalizing RGB to [0, 1] for HSV conversion
    img_norm = img_array.astype(float) / 255.0

    # Converting image to HSV
    hsv_image = mcolors.rgb_to_hsv(img_norm)

    # Storing Hue and Saturation values for the bright pixels
    h_vals = hsv_image[ys, xs, 0]
    s_vals = hsv_image[ys, xs, 1]

    # Storing the feature vector [Hue, Saturation]
    features = np.stack([h_vals, s_vals], axis=1)

    # Coordinates for plotting
    coords = np.stack([xs, ys], axis=1)

    return features, coords, hsv_image

def run_kmeans_on_hsv(
        img_array: np.ndarray,
        grey_threshold: float = 230.0,
        n_clusters: int = 3,
        random_state: int = 0,
):
    """
    Takes an RGB image, encodes it to HSV using the encode_hsv_color function, and uses K-means
    to cluster those color features

    Parameters:
        img_array : np.ndarray
        RGB image array of shape
        grey_threshold : float
            Greyscale threshold. Only pixels with grey > threshold are used.
        n_clusters : int
            Number of clusters K for K-means.
        random_state : int
            Random seed for reproducibility.

    Returns:
        labels : np.ndarray
            Cluster label for each bright pixel.
        coords : np.ndarray
            Pixel coordinates for each bright pixel.
        features : np.ndarray
            HSV features (Hue, Saturation) used by K-means.
        hsv_image : np.ndarray
            Full HSV image.
        kmeans_model : KMeans
            Fitted KMeans model.
    """
    # Encoding into HSV features (H, S) for bright pixels
    features, coords, hsv_image = encode_hsv_color(
        img_array,
        grey_threshold=grey_threshold,
    )

    # 2) Cluster the HSV features with K-means
    labels, kmeans_model = kmeans_cluster_features(
        features,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    return labels, coords, features, hsv_image, kmeans_model

# Task 7: RGB color encoding

def encode_rgb_color(
    img_array: np.ndarray,
    grey_threshold: float = 230.0,
):
    """
    Encode an RGB image directly in RGB space for bright pixels.

    We reuse the greyscale threshold to select only pixels with
    grey > grey_threshold. The feature vector is the normalized RGB values.

    Parameters
        img_array : np.ndarray
            RGB image array of shape (H, W, 3) with values in [0, 255].
        grey_threshold : float
            Greyscale threshold (on 0-255 scale). Only pixels with grey > threshold

    Returns:
        features : np.ndarray
            Array of shape (N, 3) with [R, G, B] (normalized to [0, 1])
            for each selected pixel.
        coords : np.ndarray
            Array of shape (N, 2) with (x, y) pixel coordinates of each selected
            pixel, coords[i] = [x_i, y_i].
    """
    # Compute greyscale to select bright pixels
    grey = rgb_to_greyscale(img_array)

    # Indices of bright pixels
    ys, xs = np.where(grey > grey_threshold)

    # Normalize RGB to [0, 1] for clustering
    img_norm = img_array.astype(float) / 255.0

    # Extract RGB features for the bright pixels
    r_vals = img_norm[ys, xs, 0]
    g_vals = img_norm[ys, xs, 1]
    b_vals = img_norm[ys, xs, 2]

    features = np.stack([r_vals, g_vals, b_vals], axis=1)

    # Coordinates for plotting / overlay
    coords = np.stack([xs, ys], axis=1)

    return features, coords

def run_kmeans_on_rgb(
        img_array: np.ndarray,
        grey_threshold: float = 230.0,
        n_clusters: int = 3,
        random_state: int = 0,
):
    """
    Takes an RGB image, encodes it directly in RGB space for bright pixels,
    and uses K-means to cluster those [R, G, B] color features.

    Parameters
        img_array : np.ndarray
            RGB image array of shape (H, W, 3).
        grey_threshold : float
            Greyscale threshold. Only pixels with grey > threshold are used.
        n_clusters : int
            Number of clusters K for K-means.
        random_state : int
            Random seed for reproducibility.

    Returns
        labels : np.ndarray
            Cluster label for each bright pixel.
        coords : np.ndarray
            Pixel coordinates for each bright pixel.
        features : np.ndarray
            RGB features [R, G, B] used by K-means.
        kmeans_model : KMeans
            Fitted KMeans model.
    """
    # Encode into RGB feature vectors for bright pixels
    features, coords = encode_rgb_color(
        img_array,
        grey_threshold=grey_threshold,
    )

    # Cluster the RGB features with K-means
    labels, kmeans_model = kmeans_cluster_features(
        features,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    return labels, coords, features, kmeans_model
