import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

def project_to_plane(points, normal, origin=None):
    normal = normal / np.linalg.norm(normal)             # Get the length of "normal" and divide it

    # pick a reference vector not parallel to normal
    ref = np.array([1,0,0]) if abs(normal[0]) < 0.9 else np.array([0,1,0])
    axis1 = np.cross(normal, ref); axis1 /= np.linalg.norm(axis1)
    axis2 = np.cross(normal, axis1)

    if origin is None:
        origin = points.mean(axis=0)

    rel_points = points - origin
    np.vstack((axis1, axis2))                                 # (2*3)
    coords = np.dot(rel_points, np.vstack((axis1, axis2)).T)  # N×2

    return coords, axis1, axis2, origin

def plot_velocity_contour(
    points,                     # (N,2) or (N,3) array, coordinates in meters
    vel,                        # (N,) velocity magnitude (same order as points)
    grid_res=200,               # resolution for interpolation grid (grid_res x grid_res)
    smooth_sigma=None,          # None or float sigma for gaussian_filter
    cmap="viridis",
    levels=50,
    fill_nan_method="nearest",  # method to fill NaN from cubic interpolation
    figsize=(8,4),
):
    """
    Draw a velocity contour from scattered points.
    - points: np.ndarray (N,2) or (N,3). Units: meters.
    - vel: np.ndarray (N,)
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    pts = np.asarray(points)
    vel = np.asarray(vel)

    if pts.ndim != 2 or pts.shape[0] != vel.shape[0]:
        raise ValueError("points must be (N,2) or (N,3) and vel length must match N")

    pts_x = pts[:, 0]
    pts_y = pts[:, 1]

    grid_res = 200
    # build grid
    xi = np.linspace(pts_x.min(), pts_x.max(), grid_res)
    yi = np.linspace(pts_y.min(), pts_y.max(), grid_res)
    X, Y = np.meshgrid(xi, yi)

    # cubic interpolation (smooth), may produce NaNs where extrapolation needed
    V = griddata(pts, vel, (X, Y), method='cubic')

    # fill NaNs from a more robust method (nearest) where cubic failed
    if np.any(np.isnan(V)):
        V_nearest = griddata(pts, vel, (X, Y), method=fill_nan_method)
        V = np.where(np.isnan(V), V_nearest, V)

    # optional gaussian smoothing
    if smooth_sigma is not None and smooth_sigma > 0:
        V = gaussian_filter(V, sigma=smooth_sigma, mode='nearest')

    # prepare figure
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    cf = ax.contourf(X, Y, V, levels=levels, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.02, aspect=30)
    cbar.set_label("Velocity magnitude (m/s)", fontsize=10)  # 或者 11、12


    # axes labels and aspect
    ax.set_xlabel("X (mm)" )
    ax.set_ylabel("Y (mm)")
    ax.set_title("Velocity contour")
    ax.set_aspect('equal', adjustable='box')

    # scatter original points optionally (small and semi-transparent, helpful to see sampling)
    ax.scatter(pts[:,0], pts[:,1], s=8, c='k', alpha=0.15)

    return fig, ax

