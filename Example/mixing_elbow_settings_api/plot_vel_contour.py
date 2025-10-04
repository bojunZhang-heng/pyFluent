import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker

cwd = os.getcwd()
pts = np.loadtxt(os.path.join(cwd, "pts.txt"))
np.multiply(pts, 1000, out=pts)
vel_mag = np.loadtxt(os.path.join(cwd, "vel_mag.txt"))

grid_res=200               # resolution for interpolation grid (grid_res x grid_res)
smooth_sigma=0.5          # None or float sigma for gaussian_filter
cmap="viridis"
levels=50
fill_nan_method="nearest"  # method to fill NaN from cubic interpolation
figsize=(8,4)

plt.rcParams['font.family'] = 'Times New Roman'

vel = np.asarray(vel_mag)

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
cbar.set_label("Velocity magnitude (m/s)", fontsize=7)  
cbar.ax.tick_params(labelsize=7)                # 刻度字体大小


# axes labels and aspect
ax.set_xlabel("X (mm)",fontsize=8)
ax.set_ylabel("Y (mm)",fontsize=8)
ax.tick_params(axis='both', labelsize=7)
ax.set_title("Velocity contour", fontsize=10)
ax.set_aspect('equal', adjustable='box')

# scatter original points optionally (small and semi-transparent, helpful to see sampling)
ax.scatter(pts[:,0], pts[:,1], s=4, c='k', alpha=0.12, marker='.', linewidth=0)

save_dir = os.path.join(cwd, "figure")
save_path = os.path.join(save_dir, "tmp_1.png")

fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)
