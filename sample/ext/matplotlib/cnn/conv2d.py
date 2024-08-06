import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def conv2d(input_matrix, kernel, stride=1, padding=0):
    # 获取输入矩阵和卷积核的维度
    input_dim = input_matrix.shape[0]
    kernel_dim = kernel.shape[0]

    # 计算输出矩阵的维度
    output_dim = int((input_dim - kernel_dim + 2 * padding) / stride) + 1

    # 初始化输出矩阵
    output_matrix = np.zeros((output_dim, output_dim))

    # 添加 padding
    if padding > 0:
        input_matrix = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')

    # 执行卷积操作
    for i in range(0, input_dim - kernel_dim + 1, stride):
        for j in range(0, input_dim - kernel_dim + 1, stride):
            output_matrix[i // stride, j // stride] = np.sum(input_matrix[i:i+kernel_dim, j:j+kernel_dim] * kernel)
    return output_matrix


def plot_heatmap(
    ax, title, data, vmin, vmax, fmt,
    cmap='cividis', annot=True, cbar=False, square=True,
):
    ax.axis('off')
    ax.set_title(title)
    sns.heatmap(data, vmin=vmin, vmax=vmax, annot=True, fmt=fmt, linewidth=.5, cmap=cmap, cbar=False, square=True, ax=ax)


def plot_matrixes(input_matrix, kernel, output_matrix, vmin, vmax, fmt, title='Convolution (Matrix)'):
    # 设置图表风格
    sns.set(style='whitegrid', context='talk')

    # 创建子图
    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [6, 3, 4]})
    fig.suptitle(title)

    # 绘制输入矩阵、卷积核、输出矩阵
    plot_heatmap(axes[0], 'Input', input_matrix, vmin, vmax, fmt)
    plot_heatmap(axes[1], 'Kernel', kernel, vmin, vmax, fmt)
    plot_heatmap(axes[2], 'Output', output_matrix, vmin, vmax, fmt)

    # 在子图之间添加运算符
    fig.text(1.06, 0.5, '*', fontsize=20, va='center', ha='center', transform = axes[0].transAxes, color ='black')
    fig.text(1.11, 0.5, '=', fontsize=20, va='center', ha='center', transform = axes[1].transAxes, color ='black')

    # 显示图形
    plt.tight_layout()
    plt.show()


def sample_vertical_edge_detect_1():
    # 定义输入矩阵
    input_matrix = np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
    ])

    # 定义卷积核
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])

    # 执行卷积操作
    output_matrix = conv2d(input_matrix, kernel)

    # 显示图表
    plot_matrixes(input_matrix, kernel, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Convolution (vertical+)')


def sample_vertical_edge_detect_2():
    # 定义输入矩阵
    input_matrix = np.array([
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
    ])

    # 定义卷积核
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])

    # 执行卷积操作
    output_matrix = conv2d(input_matrix, kernel)

    # 显示图表
    plot_matrixes(input_matrix, kernel, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Convolution (vertical-)')


def sample_horizontal_edge_detect_1():
    # 定义输入矩阵
    input_matrix = np.array([
        [10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    # 定义卷积核
    kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ])

    # 执行卷积操作
    output_matrix = conv2d(input_matrix, kernel)

    # 显示图表
    plot_matrixes(input_matrix, kernel, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Convolution (horizontal+)')


def sample_horizontal_edge_detect_2():
    # 定义输入矩阵
    input_matrix = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10],
    ])

    # 定义卷积核
    kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ])

    # 执行卷积操作
    output_matrix = conv2d(input_matrix, kernel)

    # 显示图表
    plot_matrixes(input_matrix, kernel, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Convolution (horizontal-)')


def sample_vertical_and_horizontal_edge_detect_1():
    # 定义输入矩阵
    input_matrix = np.array([
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
    ])

    # 定义卷积核
    kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ])

    # 执行卷积操作
    output_matrix = conv2d(input_matrix, kernel)

    # 显示图表
    plot_matrixes(input_matrix, kernel, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Convolution (horizontal+-)')


def sample_vertical_and_horizontal_edge_detect_2():
    # 定义输入矩阵
    input_matrix = np.array([
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
        [0, 0, 0, 10, 10, 10],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
    ])

    # 定义卷积核
    kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ])

    # 执行卷积操作
    output_matrix = conv2d(input_matrix, kernel)

    # 显示图表
    plot_matrixes(input_matrix, kernel, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Convolution (horizontal-+)')


def sample_vertical_and_horizontal_padding_edge_detect_1():
    # 定义输入矩阵
    input_matrix = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 10, 10, 10, 0, 0, 0, 0],
        [0, 10, 10, 10, 0, 0, 0, 0],
        [0, 10, 10, 10, 0, 0, 0, 0],
        [0, 0, 0, 0, 10, 10, 10, 0],
        [0, 0, 0, 0, 10, 10, 10, 0],
        [0, 0, 0, 0, 10, 10, 10, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])

    # 定义卷积核
    kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ])

    # 执行卷积操作
    output_matrix = conv2d(input_matrix, kernel)

    # 显示图表
    plot_matrixes(input_matrix, kernel, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Convolution (horizontal padding+-)')


def sample_vertical_and_horizontal_stride_edge_detect_1():
    # 定义输入矩阵
    input_matrix = np.array([
        [10, 10, 10, 10, 0, 0, 0],
        [10, 10, 10, 10, 0, 0, 0],
        [10, 10, 10, 10, 0, 0, 0],
        [0, 0, 0, 0, 10, 10, 10],
        [0, 0, 0, 0, 10, 10, 10],
        [0, 0, 0, 0, 10, 10, 10],
        [0, 0, 0, 0, 10, 10, 10],
    ])

    # 定义卷积核
    kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ])

    # 执行卷积操作
    output_matrix = conv2d(input_matrix, kernel, stride=2)

    # 显示图表
    plot_matrixes(input_matrix, kernel, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Convolution (horizontal stride+-)')


def sample_vertical_and_horizontal_padding_stride_edge_detect_1():
    # 定义输入矩阵
    input_matrix = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 10, 10, 10, 0, 0, 0, 0],
        [0, 10, 10, 10, 0, 0, 0, 0],
        [0, 10, 10, 10, 0, 0, 0, 0],
        [0, 0, 0, 0, 10, 10, 10, 0],
        [0, 0, 0, 0, 10, 10, 10, 0],
        [0, 0, 0, 0, 10, 10, 10, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])

    # 定义卷积核
    kernel = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1],
    ])

    # 执行卷积操作
    output_matrix = conv2d(input_matrix, kernel, stride=2)

    # 显示图表
    plot_matrixes(input_matrix, kernel, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Convolution (horizontal padding+-)')




if __name__ == "__main__":
    #sample_vertical_edge_detect_1()
    #sample_vertical_edge_detect_2()
    #sample_horizontal_edge_detect_1()
    #sample_horizontal_edge_detect_2()
    #sample_vertical_and_horizontal_edge_detect_1()
    #sample_vertical_and_horizontal_edge_detect_2()
    #sample_vertical_and_horizontal_padding_edge_detect_1()
    #sample_vertical_and_horizontal_stride_edge_detect_1()
    sample_vertical_and_horizontal_padding_stride_edge_detect_1()