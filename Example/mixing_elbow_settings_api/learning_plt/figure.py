
###############################################################################
# prepare figure
# ~~~~~~~~~~~~~~
#

fig, ax = plt.subplots(figsize=figsize)
cf = ax.contourf(X, Y, V, levels=levels, cmap=cmap)
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label("Velocity magnitude (m/s)")

# plt.subplots: 创建 figure 与 axes，大小由 figsize 决定。  
# ax.contourf: 用 X,Y,V 画填色等值线图（colormap 由 cmap，等值面数量由 levels 控制）。
#  返回的 cf 是 QuadContourSet
# fig.colorbar: 添加颜色条（legend），并标注单位（m/s

 
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Velocity contour")
ax.set_aspect('equal', adjustable='box')
# set_aspect('equal') 确保 X、Y 单位长度相同（避免图被拉伸）
# adjustable='box' 允许调整绘图区大小以保持比例。







