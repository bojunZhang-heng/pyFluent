import numpy as np

###############################################################################
# axis introduction
# ~~~~~~~~~~~~~~~~~
#

# 一维数组: 只有 axis=0
arr_1d = np.array([1, 2, 3, 4, 5])
# 轴: 0 → [1, 2, 3, 4, 5]

# 二维数组: axis=0(行), axis=1(列)
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])
# axis=0 ↓  axis=1 →
#        [1, 2, 3]
#        [4, 5, 6]

# 三维数组: axis=0, axis=1, axis=2
arr_3d = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])

###############################################################################
# build grid
# ~~~~~~~~~~
# X 保存的是每个网格点对应的 x 坐标（在每一行重复 xi）
# Y 保存的是每个网格点对应的 y 坐标（在每一列重复 yi）。
#

xi = np.linspace(pts_x.min(), pts_x.max(), grid_res)
yi = np.linspace(pts_y.min(), pts_y.max(), grid_res)
X, Y = np.meshgrid(xi, yi)

# Example
xi = np.array([0.0, 1.0, 2.0])   # nx = 3
yi = np.array([10.0, 20.0])      # ny = 2

X, Y = np.meshgrid(xi, yi)       # 默认 indexing='xy'
print(X.shape, Y.shape)          # (2, 3) (2, 3)

print(X)
# [[0. 1. 2.]
#  [0. 1. 2.]]
print(Y)
# [[10. 10. 10.]
#  [20. 20. 20.]]
