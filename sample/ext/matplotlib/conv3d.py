import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def conv3d(image, kernel, padding=0, stride=1):
    """
    Performs a 3D convolution on a 3D image with a given 3D kernel.

    :param image: Input 3D image (numpy array of shape [D, H, W]).
    :param kernel: 3D kernel (numpy array of shape [kD, kH, kW]).
    :param padding: Amount of zero-padding around the border (default: 0).
    :param stride: Stride of the convolution (default: 1).
    :return: Output 2D image after convolution.
    """
    d, h, w = image.shape
    kd, kh, kw = kernel.shape

    # Apply padding to the input image
    if padding > 0:
        image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)

    # Calculate the dimensions of the output
    output_h = (h - kh + 2 * padding) // stride + 1
    output_w = (w - kw + 2 * padding) // stride + 1

    # Initialize the output array
    output = np.zeros((output_h, output_w))

    # Perform the convolution
    for y in range(output_h):
        for x in range(output_w):
            y_start = y * stride
            x_start = x * stride
            output[y, x] = np.sum(image[:, y_start:y_start+kh, x_start:x_start+kw] * kernel)
    return output


def plot_heatmap(
    ax, title, data, vmin, vmax, fmt,
    cmap='cividis', annot=True, cbar=False, square=True,
):
    ax.axis('off')
    ax.set_title(title)
    sns.heatmap(data, vmin=vmin, vmax=vmax, annot=True, fmt=fmt, linewidth=.5, cmap=cmap, cbar=False, square=True, ax=ax)


def plot_heatmap3d_surface(ax, data, z_offset, title):
    x, y = np.meshgrid(np.arange(data.shape[1] + 1), np.arange(data.shape[0] + 1))
    z = np.full_like(x, z_offset)

    # 绘制热图
    ax.plot_surface(x, y, z, facecolors=sns.color_palette("viridis", as_cmap=True)(data), shade=False, vmin=-30, vmax=30, alpha=0.6)

    # 在平面上标注数值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j + 0.5, i + 0.5, z_offset, f'{data[i, j]:.0f}', ha='center', va='center', color='black')

    # 在每个通道的热图中心位置添加通道名称
    center_x, center_y = data.shape[1] // 2, data.shape[0] // 2
    ax.text(center_x, center_y, z_offset + 0.1, title, ha='center', va='center', color='black', fontsize=12, weight='bold')


def plot_heatmap3d(
    ax, title, data, vmin, vmax, fmt,
    cmap='cividis', annot=True, cbar=False, square=True,
):
    for i in range(3):
        plot_heatmap3d_surface(ax, data[i], i * 100, f'Channel {i+1}')

    # 设置 3D 图标题
    ax.set_title('RGB ' + title)
    ax.set_axis_off()


def plot_matrixes(image, kernel, output, vmin, vmax, fmt, title='Conv3d'):
    # 设置图表风格
    sns.set(style='whitegrid', context='talk')

    # 创建子图
    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [6, 3, 4]})
    fig.suptitle(title)

    # 绘制 3D 热图
    ax0 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2 = fig.add_subplot(1, 3, 3)

    # 绘制输入矩阵、卷积核、输出矩阵
    plot_heatmap3d(ax0, 'Input', image, vmin, vmax, fmt)
    plot_heatmap3d(ax1, 'Kernel', kernel, vmin, vmax, fmt)
    plot_heatmap(ax2, 'Output', output, vmin, vmax, fmt)

    # 在子图之间添加运算符
    fig.text(1.06, 0.5, '*', fontsize=20, va='center', ha='center', transform = axes[0].transAxes, color ='black')
    fig.text(1.11, 0.5, '=', fontsize=20, va='center', ha='center', transform = axes[1].transAxes, color ='black')

    # 显示图形
    plt.tight_layout()
    plt.show()


def sample_vertical3d_1():
    # 定义一个 RGB 图像（6x6x3）
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

    # 定义卷积核（3x3x3）
    kernel = np.array([
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ],
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ],
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ],
    ])

    # 对 RGB 图像应用 3D 卷积
    output = conv3d(image, kernel, padding=1, stride=1)
    print("Output Shape:", output.shape)
    print("Output:\n", output)

    # 显示图表
    plot_matrixes(image, kernel, output, vmin=-30, vmax=30, fmt='.0f', title='Convolution (vertical3d+)')




def aaa():
    # 生成一个随机 RGB 图像（例如 5x5x3）
    image = np.random.rand(3, 5, 5)  # (D, H, W)

    # 定义一个 3x3x3 卷积核
    kernel = np.ones((3, 3, 3)) / 27  # 简单的平均值卷积核
    print(image, image.shape, image[:, :, 0].shape)
    print(kernel, kernel.shape)

    # 对 RGB 图像应用 3D 卷积
    output = conv3d(image, kernel, padding=1, stride=1)
    print("Output Shape:", output.shape)
    print("Output:\n", output)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


    # 绘制3D立方体并标注数值
    def plot_cube_with_values(ax, image, position, size):
        d, h, w = image.shape
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    # 绘制立方体
                    r = [0, size]
                    X, Y = np.meshgrid(r, r)
                    one = np.ones_like(X)
                    ax.plot_surface(X + x * size + position[0], Y + y * size + position[1], one * size + z * size + position[2], color='blue', alpha=0.3)
                    ax.plot_surface(X + x * size + position[0], Y + y * size + position[1], one * 0 + z * size + position[2], color='blue', alpha=0.3)
                    ax.plot_surface(X + x * size + position[0], one * size + y * size + position[1], Y + z * size + position[2], color='blue', alpha=0.3)
                    ax.plot_surface(X + x * size + position[0], one * 0 + y * size + position[1], Y + z * size + position[2], color='blue', alpha=0.3)
                    ax.plot_surface(one * size + x * size + position[0], X + y * size + position[1], Y + z * size + position[2], color='blue', alpha=0.3)
                    ax.plot_surface(one * 0 + x * size + position[0], X + y * size + position[1], Y + z * size + position[2], color='blue', alpha=0.3)
                    # 标注数值
                    ax.text(x * size + position[0] + size / 2, y * size + position[1] + size / 2, z * size + position[2] + size / 2, f'{image[z, y, x]:.2f}', ha='center', va='center', color='black')

    # 创建图形和3D轴
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制输入图像的3D立方体
    plot_cube_with_values(ax, image, position=[0, 0, 0], size=1)

    # 设置轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Input Image with Values')


    # 绘制函数
    def plot_with_values(ax, matrix, title):
        cax = ax.matshow(matrix, cmap='viridis')
        for (i, j), val in np.ndenumerate(matrix):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')
        ax.set_title(title)
        plt.colorbar(cax, ax=ax)

    # 创建图形和子图
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # 绘制输入图像的每个通道
    for i in range(3):
        plot_with_values(axs[0, i], image[i], f'Input Channel {i+1}')

    # 绘制卷积核的平面投影
    plot_with_values(axs[1, 0], np.sum(kernel, axis=0), 'Kernel Projection (Z-Sum)')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    # 绘制输出图像
    plot_with_values(axs[1, 1], output, 'Output Image')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    # 关闭未用的子图
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    sample_vertical3d_1()

