import matplotlib.pyplot as plt

#----------------------- colormap -----------------------
# "viridis"（默认，蓝→绿→黄）  
#

cmp = ["viridis", "jet",  "cividis"]



#----------------------- clamp -----------------------
# clamp
# ~~~~~
# clamp,将取值限定在(-2, 2)
# target[:,0] : 仅考虑第二维中的第一个分量
#   (N,4)  -> (N,1)
#

targets["volume_anchor_velocity"].cpu()[:, 0].clamp(-2, 2)

###############################################################################
#----------------------- fig setting -----------------------
#

fig = []
fig = plt.figure(figsize=figsize)

# i = 0 1 2 使用三位数子图编号法（1 3 k）构建 1 行 3 列的
fig.add_subplot(130 + i + 1, projection="3d")

###############################################################################
#----------------------- axs -----------------------
# 准备收集每个子图的 axis 对象
#

axs = []
ax = fig.add_subplot(130 + i + 1, projection="3d")
axs.append(ax)：把 axis 存起来以便 later 使用（比如 colorbar）。

# 设置标g（title）字体
ax.title.set_fontsize(9)
ax.title.set_fontname("Times New Roman")

# 设置坐标轴名字（X/Y/Z label）字体
ax.xaxis.label.set_fontsize(9)
ax.xaxis.label.set_fontname("Times New Roman")

ax.yaxis.label.set_fontsize(9)
ax.yaxis.label.set_fontname("Times New Roman")

ax.zaxis.label.set_fontsize(9)
ax.zaxis.label.set_fontname("Times New Roman")

# 设置坐标轴刻度标签（tick labels）字体
for tick in ax.get_xticklabels():
    tick.set_fontsize(9)
    tick.set_fontname("Times New Roman")

for tick in ax.get_yticklabels():
    tick.set_fontsize(9)
    tick.set_fontname("Times New Roman")

for tick in ax.get_zticklabels():
    tick.set_fontsize(9)
    tick.set_fontname("Times New Roman")

# 如果你还想让 tick 更稀疏（刻度别太密）
ax.xaxis.set_major_locator(plt.MaxNLocator(4))
ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.zaxis.set_major_locator(plt.MaxNLocator(4))

# 如果你想让刻度线变细、变优雅
ax.tick_params(width=0.5, pad=2)

# Set box aspect based one actual data
data_x_range = x.max() - x.min()
data_y_range = y.max() - y.min()
data_z_range = z.max() - z.min()
axs[i].set_box_aspect((float(data_x_range), float(data_y_range), float(data_z_range)))

#----------------------- unbind -----------------------
#unbind(-1)：把最后一维“拆开”为多个 tensor
#

pos[i].shape == (N, 3)
x, y, z = pos[i][perm].unbind(-1)]
x = [x1, x2, ..., xN]
y = [y1, y2, ..., yN]
z = [z1, z2, ..., zN]

#----------------------- scatter -----------------------
# 散点图绘制
# x, y, z 表示三个方向的数据信息
# s=3 表示点的大小
# c=delta[perm] 表示color的取值范围, 颜色区间
# alpha=alpha 表示透明度
#

scatters = []
scatter = ax.scatter(
    x, y, z, s=3,
    c=delta[perm] if is_delta else color[i][perm],
    cmap="coolwarm",
    vmin=None if is_delta else vmin,
    vmax=None if is_delta else vmax,
    alpha=alpha,
)

