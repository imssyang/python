import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 生成示例数据
image = np.array([
    [
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
    ],
    [
        [10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    [
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
    ]
])
print(image.shape, image)

# 创建图形和3D轴
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')


# 用于绘制heatmap的函数
def plot_heatmap_on_cube(ax, data, z_offset, title):
    x, y = np.meshgrid(np.arange(data.shape[1] + 1), np.arange(data.shape[0] + 1))
    z = np.full_like(x, z_offset)

    ax.plot_surface(x, y, z, facecolors=sns.color_palette("viridis", as_cmap=True)(data), shade=False, vmin=-30, vmax=30, alpha=0.6)

    # 在平面上标注数值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j + 0.5, i + 0.5, z_offset, f'{data[i, j]:.0f}', ha='center', va='center', color='black')

    # 在每个通道的热图中心位置添加通道名称
    center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
    ax.text(center_x, center_y, z_offset + 0.1, title, ha='center', va='center', color='black', fontsize=12, weight='bold')

# 绘制每个通道的热图到3D轴上
for i in range(3):
    plot_heatmap_on_cube(ax, image[i], i * 100, f'Channel {i+1}')

# 设置轴标签和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Heatmap of RGB Channels')
ax.set_axis_off()


plt.tight_layout()
plt.show()