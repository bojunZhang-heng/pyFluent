import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

###############################################################################
# 使用 scipy.interpolate.griddata 将散点 (xy, vel) 插值到规则网格 (X, Y) 上
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Explanation: 使用 cubic（立方）插值，返回 V（grid_res × grid_res）表示每个网格点的速度估计值。
# Remark: cubic 在二维散点上可能较慢、且在网格外或稀疏区域会产生 NaN（因为这是插值非外推）。
#         若数据非常不规则/稀疏，可考虑 method='linear' 或 'nearest'。

V = griddata(xy, vel, (X, Y), method='cubic')




###############################################################################
# fill NaNs from a more robust method (nearest) where cubic failed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

if np.any(np.isnan(V)):
    V_nearest = griddata(pts, vel, (X, Y), method='linear')
    V = np.where(np.isnan(V), V_nearest, V)

# np.isnan() 
# 对数组 V 做元素级判断
# return：与 V 同形状的布尔数组，True 表示该位置是 NaN，False 表示不是 

# np.any()
# 判断输入布尔数组中是否至少有一个元素为 True。
# Return：单个布尔值 True 或 False。

# np.where(np.isnan(V), V_nearest, V)
# 按条件从两个数组中逐元素选择值：
# 若条件为 True，取第一个数组的对应元素；否则取第二个数组的对应元素


###############################################################################
# Gaussian smoothen
# ~~~~~~~~~~~~~~~~~
# smooth_sigma commonly chooses 0.5 - 2 
#

if smooth_sigma is not None and smooth_sigma > 0:
    V = gaussian_filter(V, sigma=smooth_sigma, mode='nearest')

# 平滑会降低噪音但也会模糊细节，依据数据与目的调整 sigma。

