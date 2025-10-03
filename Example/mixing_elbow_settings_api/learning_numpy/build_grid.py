import numpy as np

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

