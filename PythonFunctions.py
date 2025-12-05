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

# Topic 2

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def set_global_seed(seed=1):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Task 0 - Plots from project 2
def load_ebola_series(path, step_days=7):
    """
    Load a whitespace-delimited WHO .dat file where the 3rd column = new cases.
    Returns time in *days since first nonzero entry*, new cases, and cumulative.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    new = pd.to_numeric(df.iloc[:, 2], errors="coerce").astype(float).values

    # start clock at the first nonzero entry
    nz = np.nonzero(new > 0)[0]
    start = nz[0] if len(nz) else 0
    new = new[start:]
    t_days = np.arange(len(new)) * step_days
    cum = np.cumsum(new)
    return t_days, new, cum

def plot_country(path, country, step_days=7):
    t, new, cum = load_ebola_series(path, step_days)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Left axis: NEW cases as red open circles (scatter)
    ax.scatter(t, new, facecolors="none", edgecolors="red", s=36, linewidths=1.2, label="New outbreaks")
    ax.set_xlabel("Days since first outbreak")
    ax.set_ylabel("Number of outbreaks")
    ax.grid(True, alpha=0.3)

    # Right axis: CUMULATIVE as black squares + connecting line
    ax2 = ax.twinx()
    ax2.plot(t, cum, color="black", linewidth=1.5)
    ax2.plot(t, cum, linestyle="None", marker="s", markersize=4, markerfacecolor="none",
             markeredgecolor="black", label="Cumulative number of outbreaks")
    ax2.set_ylabel("Cumulative number of outbreaks")

    # Title + combined legend
    ax.set_title(f"Ebola outbreaks in {country}")
    lines, labels = [], []
    for a in (ax, ax2):
        L = a.get_legend_handles_labels()
        lines += L[0]; labels += L[1]
    ax.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.show()

    # Quick summary in the console
    print(f"{country}: total cumulative = {int(cum[-1])}, "
          f"peak new = {int(new.max())} at day ≈ {t[np.argmax(new)]:.0f}")

# Task 1

def fit_linear(t_days, y, window_days=None):
    """
    Fit y ~ a*t + b.
    Returns: (model, t_fit, y_pred, r2, rmse)
    """
    t = np.asarray(t_days, dtype=float)
    yy = np.asarray(y, dtype=float)

    if window_days is not None:
        mask = (t >= 0) & (t <= window_days)
        t = t[mask]; yy = yy[mask]

    X = t.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, yy)
    y_pred = model.predict(X)

    r2 = r2_score(yy, y_pred)
    rmse = np.sqrt(mean_squared_error(yy, y_pred))
    return model, t, y_pred, r2, rmse



def plot_with_linear_fit(path: Path, country: str, step_days=7, window_days=None):
    # Load series
    t, new, cum = load_ebola_series(path, step_days)

    # NEW cases fit
    model_new, tN, yN_pred, r2_new, rmse_new = fit_linear(t, new, window_days)

    # CUMULATIVE fit
    model_cum, tC, yC_pred, r2_cum, rmse_cum = fit_linear(t, cum, window_days)

    # Plot NEW cases
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(t, new, facecolors="none", edgecolors="red", s=30, linewidths=1.0, label="New cases (data)")
    ax.plot(tN, yN_pred, color="red", lw=2, label=f"Linear fit  (R²={r2_new:.3f}, RMSE={rmse_new:.1f})")
    ax.set_title(f"{country} — Linear regression on NEW cases"
                 + (f" (0–{window_days} d)" if window_days else " (full series)"))
    ax.set_xlabel("Days since first outbreak"); ax.set_ylabel("New cases per report")
    ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout(); plt.show()

    # Plot CUMULATIVE 
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(t, cum, facecolors="none", edgecolors="black", s=28, linewidths=1.0, label="Cumulative (data)")
    ax.plot(tC, yC_pred, color="black", lw=2, label=f"Linear fit  (R²={r2_cum:.3f}, RMSE={rmse_cum:.1f})")
    ax.set_title(f"{country} —c Linear regression on CUMULATIVE"
                 + (f" (0–{window_days} d)" if window_days else " (full series)"))
    ax.set_xlabel("Days since first outbreak"); ax.set_ylabel("Cumulative cases")
    ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout(); plt.show()

    print(f"{country} (NEW):   y ≈ {model_new.coef_[0]:.3f}·t + {model_new.intercept_:.1f}")
    print(f"{country} (CUMUL): y ≈ {model_cum.coef_[0]:.3f}·t + {model_cum.intercept_:.1f}\n")

# Task 2

def design_matrix_poly(t, degree):
    """
    Build a polynomial design matrix:
    [1, t, t^2, ..., t^degree]
    for each element in t.
    """
    t = np.asarray(t, float)
    # columns: t^0, t^1, ..., t^degree
    X_cols = [np.ones_like(t)]
    for k in range(1, degree + 1):
        X_cols.append(t**k)
    X = np.column_stack(X_cols)
    return X


def fit_poly_basic(t_days, y, degree=1):
    """
    Fit a polynomial linear regression of given degree:
         y ≈ b0 + b1*t + ... + b_degree*t^degree
    using the normal equations.
    """
    t = np.asarray(t_days, float)
    yy = np.asarray(y, float)

    # Design matrix
    X = design_matrix_poly(t, degree)

    XT_X = X.T @ X

    XT_y = X.T @ yy

    # Solve for b
    b = np.linalg.solve(XT_X, XT_y)

    # Predictions
    y_pred = X @ b

    # Metrics: RMSE and R^2
    rmse = np.sqrt(np.mean((yy - y_pred)**2))

    ss_res = np.sum((yy - y_pred)**2)
    ss_tot = np.sum((yy - np.mean(yy))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "beta": b,
        "t": t,
        "y": yy,
        "y_pred": y_pred,
        "rmse": rmse,
        "r2": r2,
    }

def run_country_poly_basic(path: Path, country: str, step_days=7, degree_new=3, degree_cum=3):
    t, new, cum = load_ebola_series(path, step_days)

    # Fit polynomial models
    res_new = fit_poly_basic(t, new, degree=degree_new)
    res_cum = fit_poly_basic(t, cum, degree=degree_cum)

    # Plot NEW
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(res_new["t"], res_new["y"], facecolors="none",
               edgecolors="red", s=28, label="New (data)")
    ax.plot(res_new["t"], res_new["y_pred"], color="red", lw=2,
            label=f"deg={degree_new}, "
                  f"(R²={res_new['r2']:.3f}, RMSE={res_new['rmse']:.1f})")
    ax.set_title(f"{country} — Polynomial fit on NEW")
    ax.set_xlabel("Days since first outbreak")
    ax.set_ylabel("New cases per report")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot CUMULATIVE
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(res_cum["t"], res_cum["y"], facecolors="none",
               edgecolors="black", s=28, label="Cumulative (data)")
    ax.plot(res_cum["t"], res_cum["y_pred"], color="black", lw=2,
            label=f"deg={degree_cum} "
                  f"(R²={res_cum['r2']:.3f}, RMSE={res_cum['rmse']:.1f})")
    ax.set_title(f"{country} — Polynomial fit on CUMULATIVE")
    ax.set_xlabel("Days since first outbreak")
    ax.set_ylabel("Cumulative cases")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"{country} NEW   -> degree={degree_new},  "
          f"R²={res_new['r2']:.3f}, RMSE={res_new['rmse']:.1f}")
    print(f"{country} CUMUL -> degree={degree_cum},  "
          f"R²={res_cum['r2']:.3f}, RMSE={res_cum['rmse']:.1f}")

# Task 3

# ---------------------------------------------------------------------
# 1) Build a small MLP for 1D time input
# ---------------------------------------------------------------------
def build_mlp(input_dim=1, hidden_units=(32, 16), lr=1e-3):
    """
    Create a simple fully-connected neural network for regression:
        input -> dense -> dense -> output(1).
    """
    model = Sequential()
    # explicit Input layer
    model.add(Input(shape=(input_dim,)))

    # hidden layers
    for h in hidden_units:
        model.add(Dense(h, activation="relu"))

    # output layer: one continuous value (cases)
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model


# ---------------------------------------------------------------------
# 2) Train NN for one country and one target ("new" or "cum")
# ---------------------------------------------------------------------
def train_nn_ebola(
    path,
    country,
    step_days=7,
    target="new",
    train_frac=0.7,
    hidden_units=(32, 16),
    lr=1e-3,
    epochs=500,
    batch_size=16,
    verbose=0,
):
    """
    Train a small NN to predict either NEW or CUMULATIVE cases
    as a function of time for a single country.

    Time-split: the first `train_frac` part of the time series is used
    for training; the rest is used as test. NO shuffling is done.
    """
    # ---- load data ----
    t_days, new, cum = load_ebola_series(path, step_days)
    if target.lower() == "new":
        y = np.asarray(new, float)
        y_label = "New cases per report"
    else:
        y = np.asarray(cum, float)
        y_label = "Cumulative cases"

    t = np.asarray(t_days, float)

    # ---- build features and scale ----
    # X: 1D time input, scaled to [0, 1]
    X = t.reshape(-1, 1)
    x_scaler = MinMaxScaler()
    X_scaled = x_scaler.fit_transform(X)

    # y: scale as well for stable NN training; we'll invert later
    y = y.reshape(-1, 1)
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)

    # ---- time-aware train / test split (no shuffling!) ----
    n = len(t)
    n_train = int(train_frac * n)
    X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]
    y_train, y_test = y_scaled[:n_train], y_scaled[n_train:]

    # ---- build and train model ----
    model = build_mlp(input_dim=1, hidden_units=hidden_units, lr=lr)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # ---- predictions on the whole series (for plotting / metrics) ----
    y_pred_scaled = model.predict(X_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = y.flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    result = {
        "country": country,
        "target": target,
        "t": t,
        "y_true": y_true,
        "y_pred": y_pred,
        "model": model,
        "history": history.history,
        "rmse": rmse,
        "r2": r2,
        "train_index": n_train,
        "y_label": y_label,
    }
    return result


# ---------------------------------------------------------------------
# 3) Helper to plot NN fit, highlighting train vs test region
# ---------------------------------------------------------------------
def plot_nn_result(res):
    """
    Plot data vs NN prediction for one country/target.
    Marks the train and test regions along the time axis.
    """
    t = res["t"]
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    n_train = res["train_index"]

    t_train = t[:n_train]
    t_test = t[n_train:]

    fig, ax = plt.subplots(figsize=(8, 4))

    # training data
    ax.scatter(t_train, y_true[:n_train], s=20, facecolors="none",
               edgecolors="tab:blue", label="Train data")

    # test data
    ax.scatter(t_test, y_true[n_train:], s=20, facecolors="none",
               edgecolors="tab:orange", label="Test data")

    # NN prediction on full time axis
    ax.plot(t, y_pred, color="red", linewidth=2,
            label=f"NN prediction (R²={res['r2']:.3f}, RMSE={res['rmse']:.1f})")

    ax.set_xlabel("Days since first outbreak")
    ax.set_ylabel(res["y_label"])
    ax.set_title(f"{res['country']} — NN fit on {res['target'].upper()} cases")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

# Task 4

def make_lstm_sequences(y, seq_len):
    """
    Turn a 1D array y[0..T-1] into supervised sequences for LSTM:
    X[i] = [y[i], ..., y[i+seq_len-1]], y_target[i] = y[i+seq_len]

    Parameters
    ----------
    y : (T,) array
        Time series (already scaled if you use a scaler).
    seq_len : int
        Number of past time steps to feed into the LSTM.

    Returns
    -------
    X : (T - seq_len, seq_len, 1) array
    y_target : (T - seq_len,) array
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    X_list = []
    y_list = []
    for i in range(T - seq_len):
        X_list.append(y[i : i + seq_len])
        y_list.append(y[i + seq_len])
    X = np.array(X_list)[..., np.newaxis]  # add feature dimension
    y_target = np.array(y_list)
    return X, y_target

def build_lstm_model(seq_len, n_features=1, units=50, lr=1e-3):
    """
    Build a simple LSTM regression model:
        Input(seq_len, n_features) -> LSTM(units) -> Dense(1)

    Parameters
    ----------
    seq_len : int
        Length of input sequence.
    n_features : int
        Number of features per time step (1 here: the case count).
    units : int
        Number of LSTM units.
    lr : float
        Learning rate for Adam.

    Returns
    -------
    model : compiled Keras model
    """
    model = Sequential()
    # explicit Input layer avoids Keras warnings
    model.add(Input(shape=(seq_len, n_features)))
    model.add(LSTM(units))
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model

def train_lstm_ebola(
    path,
    country,
    target="new",        # "new" or "cum"
    step_days=7,
    seq_len=5,
    train_frac=0.7,
    units=50,
    lr=1e-3,
    epochs=300,
    batch_size=16,
    verbose=0,
):
    """
    Train an LSTM on Ebola time series for one country.

    We:
      1) load the series with load_ebola_series
      2) choose target = 'new' or 'cum'
      3) scale y to [0,1] with MinMaxScaler
      4) build (X, y) sequences using make_lstm_sequences
      5) split chronologically: first train_frac for training, rest for test
      6) train the LSTM
      7) predict on the whole series and invert the scaling
      8) compute R^2 and RMSE on the test set

    Returns
    -------
    result : dict with
        'country', 'target', 't_seq', 'y_true', 'y_pred',
        't_train', 'y_train', 't_test', 'y_test',
        'r2', 'rmse', 'seq_len'
    """
    # 1) load data
    t_days, new, cum = load_ebola_series(path, step_days=step_days)
    if target == "new":
        y_raw = new
    elif target == "cum":
        y_raw = cum
    else:
        raise ValueError("target must be 'new' or 'cum'")

    y_raw = np.asarray(y_raw, dtype=float)

    # 2) scale to [0,1] 
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_raw_2d = y_raw.reshape(-1, 1)
    
    scaler.fit(y_raw_2d)
    y_scaled = scaler.transform(y_raw_2d).flatten()

    # 3) build supervised sequences
    X_all, y_all = make_lstm_sequences(y_scaled, seq_len=seq_len)
    # times corresponding to each target (we predict y[t] from previous seq_len steps)
    t_seq = t_days[seq_len:]

    # 4) chronological split
    n_total = len(y_all)
    n_train = int(train_frac * n_total)
    X_train, X_test = X_all[:n_train], X_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]
    t_train, t_test = t_seq[:n_train], t_seq[n_train:]

    # 5) build and train model
    model = build_lstm_model(seq_len=seq_len, n_features=1, units=units, lr=lr)
    es = EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=False,   
        callbacks=[es],
    )

    # 6) predictions for the whole sequence
    y_pred_scaled = model.predict(X_all, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_all.reshape(-1, 1)).flatten()

    # 7) metrics on test part only (in original units)
    y_test_pred = y_pred[n_train:]
    y_test_true = y_true[n_train:]
    r2 = r2_score(y_test_true, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

    result = {
        "country": country,
        "target": target,
        "t_seq": t_seq,
        "y_true": y_true,
        "y_pred": y_pred,
        "t_train": t_train,
        "y_train": y_true[:n_train],
        "t_test": t_test,
        "y_test": y_true[n_train:],
        "r2": r2,
        "rmse": rmse,
        "seq_len": seq_len,
        "history": history.history,
    }
    return result

def plot_lstm_result(res):
    """
    Plot LSTM predictions vs data for one country/target.
    """
    country = res["country"]
    target = res["target"]
    t_seq = res["t_seq"]
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    t_train = res["t_train"]
    y_train = res["y_train"]
    t_test = res["t_test"]
    y_test = res["y_test"]
    r2 = res["r2"]
    rmse = res["rmse"]

    ylabel = "New cases per report" if target == "new" else "Cumulative cases"

    plt.figure(figsize=(9, 4))
    plt.scatter(t_train, y_train, s=25, facecolors="none", edgecolors="C0", label="Train data")
    plt.scatter(t_test,  y_test,  s=25, facecolors="none", edgecolors="C1", label="Test data")

    plt.plot(t_seq, y_pred, "r-", label=f"LSTM prediction (R²={r2:.3f}, RMSE={rmse:.1f})")

    plt.xlabel("Days since first outbreak")
    plt.ylabel(ylabel)
    plt.title(f"{country} — LSTM fit on {target.upper()} cases")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()