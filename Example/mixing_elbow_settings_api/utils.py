import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

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


def smooth_velocity_contour_physical(coords_m, speed,
                                     outlet_size_mm=(10.0, 2.5),
                                     grid_res=(400, 100),
                                     method='cubic',
                                     gaussian_sigma=1.0,
                                     fill_method='nearest',
                                     cmap='viridis',
                                     show_scatter=False,
                                     quiver_vecs_2d=None,
                                     quiver_step=None,
                                     quiver_scale=1.0,
                                     figsize=(8,4),
                                     savepath=None):
    """
    coords_m: (N,2) coordinates on plane in meters
    speed: (N,) velocity magnitudes (any unit, e.g., m/s)
    outlet_size_mm: tuple (width_mm, height_mm) physical size of outlet in mm (10 x 2.5)
    grid_res: (nx, ny) grid resolution (higher -> smoother but slower)
    method: interpolation ('cubic'|'linear'|'nearest')
    gaussian_sigma: smoothing on the gridded field (in grid pixels)
    fill_method: 'nearest' to fill NaNs after cubic interpolation, or None
    quiver_vecs_2d: optional (N,2) projected velocity components to overlay arrows (units consistent)
    quiver_step: subsampling step for quiver (None -> automatic)
    quiver_scale: matplotlib quiver scale
    savepath: filename to save figure (png/pdf) or None
    Returns: X_mm, Y_mm, Z_smooth (grid in mm, and smoothed field)
    """

    # --- Convert coords from meters -> mm for plotting / physical sizing ---
    coords_mm = np.asarray(coords_m) * 1000.0
    x = coords_mm[:,0]
    y = coords_mm[:,1]
    w_mm, h_mm = outlet_size_mm

    # --- Define grid extent centered on the mean point, but clipped to cover data ---
    cx, cy = x.mean(), y.mean()
    half_w, half_h = w_mm/2.0, h_mm/2.0

    # grid extents (centered on centroid, using outlet physical size)
    xmin, xmax = cx - half_w, cx + half_w
    ymin, ymax = cy - half_h, cy + half_h

    # ensure grid fully covers sample points (in case coords are slightly outside)
    xmin = min(xmin, x.min())
    xmax = max(xmax, x.max())
    ymin = min(ymin, y.min())
    ymax = max(ymax, y.max())

    nx, ny = grid_res
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xi, yi)

    # --- Interpolate scattered speed onto grid ---
    Z = griddata((x, y), speed, (X, Y), method=method)

    # fill NaNs with nearest if requested
    if np.any(np.isnan(Z)) and fill_method == 'nearest':
        Z_nearest = griddata((x, y), speed, (X, Y), method='nearest')
        Z = np.where(np.isnan(Z), Z_nearest, Z)

    # Gaussian smoothing on the gridded field
    if gaussian_sigma is not None and gaussian_sigma > 0:
        # handle NaNs before smoothing: replace small remaining NaNs with local mean (simple)
        if np.any(np.isnan(Z)):
            # replace NaN with nearest fill as fallback
            Z = np.nan_to_num(Z, nan=np.nanmean(Z))
        Z_smooth = gaussian_filter(Z, sigma=gaussian_sigma, mode='nearest')
    else:
        Z_smooth = Z

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    # use pcolormesh (X,Y are in mm)
    pcm = ax.pcolormesh(X, Y, Z_smooth, shading='auto', cmap=cmap)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('Velocity magnitude')

    # optional contour lines for clarity
    try:
        cs = ax.contour(X, Y, Z_smooth, levels=8, colors='k', linewidths=0.5, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    except Exception:
        pass

    # overlay original sample points
    if show_scatter:
        ax.scatter(x, y, c='white', s=10, edgecolor='k', linewidth=0.2, alpha=0.9, zorder=3)

    # optional quiver (projected 2D velocities provided in same mm-space scaling assumption)
    if quiver_vecs_2d is not None:
        u = quiver_vecs_2d[:,0]
        v = quiver_vecs_2d[:,1]
        # if user didn't pass step, determine a reasonable subsample
        N = len(x)
        if quiver_step is None:
            quiver_step = max(1, int(N / 500))
        ax.quiver(x[::quiver_step], y[::quiver_step], u[::quiver_step], v[::quiver_step],
                  angles='xy', scale_units='xy', scale=quiver_scale, width=0.0025, zorder=4)

    # set axis labels and aspect, in mm (so physical proportions preserved)
    ax.set_xlabel('Axis 1 (mm)')
    ax.set_ylabel('Axis 2 (mm)')
    ax.set_title('Smoothed velocity magnitude (mm units on axes)')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300)
    plt.show()

    return X, Y, Z_smooth

# ---------------------------
# 使用示例（假设你已有 coords (m) 和 speed (m/s)）：
# coords_m = ...   # shape (N,2), 单位 m
# speed = ...      # shape (N,), 单位 m/s
# 若你已有 3D velocities 并做过平面投影得到 quiver_vecs (N,2)，可以传入 quiver_vecs_2d（注意单位）
#
# 示例调用（出口 10mm x 2.5mm）：
# X_mm, Y_mm, Z = smooth_velocity_contour_physical(coords_m, speed,
#                                                  outlet_size_mm=(10.0, 2.5),
#                                                  grid_res=(400, 100),
#                                                  method='cubic',
#                                                  gaussian_sigma=1.2,
#                                                  fill_method='nearest',
#                                                  show_scatter=False,
#                                                  quiver_vecs_2d=None,
#                                                  savepath='velocity_contour.png')
