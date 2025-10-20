import numpy as np

# 假设你已有的数组（示例形状）
# wall_area_mag: shape (N,)
# wall_area: shape (N,3)
# wall_p: shape (N,)


cosines = wall_area / wall_area_mag[:, None]         # shape (N,3)
# [:, None]: 将一维[N,] 变成二维[N, 1]
# 在计算时， Numpy 会把[N, 1] 沿列扩展为[N, 3]

angles_rad = np.arccos(cosines)                      # shape (N,3)
angles_deg = np.degrees(angles_rad)                  # shape (N,3)
