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


def plot_heatmap3d_surface(ax, data, z_offset, cmap, ch_title, ax_title):
    x, y = np.meshgrid(np.arange(data.shape[1] + 1), np.arange(data.shape[0] + 1))
    z = np.full_like(x, z_offset)

    # 绘制热图
    ax.plot_surface(x, y, z, facecolors=sns.color_palette(cmap, as_cmap=True)(data), shade=False, vmin=-60, vmax=60, alpha=0.5)

    # 在平面上标注数值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j + 0.5, i + 0.5, z_offset, f'{data[i, j]:.0f}', ha='center', va='center', color='black')

    # 在每个通道的热图中心位置添加通道名称
    center_x, center_y = data.shape[1] // (2 if ax_title == 'Kernel' else 4), data.shape[0] // 2
    ax.text(-center_x, center_y, z_offset + 0.1, ch_title, ha='left', va='center', color='Blue', fontsize=12, weight='bold')


def plot_heatmap3d(
    ax, title, data, vmin, vmax, fmt,
    cmap='cividis', annot=True, cbar=False, square=True,
):
    ax.set_title(title)
    ax.set_axis_off()
    channle_names = {0: 'Red', 1: 'Green', 2: 'Blue'}
    for i in range(3):
        plot_heatmap3d_surface(ax, data[i], i * 100, cmap, channle_names[i], title)


def plot_conv3d(image, kernel, output, vmin, vmax, fmt, title='Conv3d'):
    # 设置图表风格
    sns.set(style='whitegrid', context='talk')

    # 创建子图
    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [10, 5, 4]})
    fig.suptitle(title)

    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 300, pos.height * 300])
        ax.axis('off')

    # 绘制 3D 热图
    ax0 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2 = fig.add_subplot(1, 3, 3)

    # 绘制输入矩阵、卷积核、输出矩阵
    plot_heatmap3d(ax0, 'Input', image, vmin, vmax, fmt)
    plot_heatmap3d(ax1, 'Kernel', kernel, vmin, vmax, fmt)
    plot_heatmap(ax2, 'Output', output, vmin, vmax, fmt)

    # 在子图之间添加运算符
    fig.text(1.06, 0.5, '*', fontsize=30, va='center', ha='right', transform = axes[0].transAxes, color ='black')
    fig.text(1.11, 0.5, '=', fontsize=30, va='center', ha='right', transform = axes[1].transAxes, color ='black')

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
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ])

    # 对 RGB 图像应用 3D 卷积
    output = conv3d(image, kernel, padding=0, stride=1)
    print("Output Shape:", output.shape)
    print("Output:\n", output)

    # 显示图表
    plot_conv3d(image, kernel, output, vmin=-60, vmax=60, fmt='.0f', title='Convolution (vertical3d+)')


def sample_vertical3d_2():
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
    output = conv3d(image, kernel, padding=0, stride=1)
    print("Output Shape:", output.shape)
    print("Output:\n", output)

    # 显示图表
    plot_conv3d(image, kernel, output, vmin=-60, vmax=60, fmt='.0f', title='Convolution (vertical3d+)')


def sample_horizontal3d_1():
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
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ])

    # 对 RGB 图像应用 3D 卷积
    output = conv3d(image, kernel, padding=0, stride=1)
    print("Output Shape:", output.shape)
    print("Output:\n", output)

    # 显示图表
    plot_conv3d(image, kernel, output, vmin=-60, vmax=60, fmt='.0f', title='Convolution (horizontal3d+)')


def sample_horizontal3d_2():
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
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
        ],
        [
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
        ],
        [
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
        ],
    ])

    # 对 RGB 图像应用 3D 卷积
    output = conv3d(image, kernel, padding=0, stride=1)
    print("Output Shape:", output.shape)
    print("Output:\n", output)

    # 显示图表
    plot_conv3d(image, kernel, output, vmin=-60, vmax=60, fmt='.0f', title='Convolution (horizontal3d+)')


def sample_vertical_and_horizontal3d_1():
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
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ])

    # 对 RGB 图像应用 3D 卷积
    output = conv3d(image, kernel, padding=0, stride=1)
    print("Output Shape:", output.shape)
    print("Output:\n", output)

    # 显示图表
    plot_conv3d(image, kernel, output, vmin=-60, vmax=60, fmt='.0f', title='Convolution (vertical3d+)')


if __name__ == "__main__":
    #sample_vertical3d_1()
    sample_vertical3d_2()
    #sample_horizontal3d_1()
    #sample_horizontal3d_2()
    #sample_vertical_and_horizontal3d_1()

