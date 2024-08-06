import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pooling(input_matrix, pool_size, stride, mode='max'):
    output_shape = ((input_matrix.shape[0] - pool_size) // stride + 1,
                    (input_matrix.shape[1] - pool_size) // stride + 1)
    pooled_matrix = np.zeros(output_shape)
    for i in range(0, input_matrix.shape[0] - pool_size + 1, stride):
        for j in range(0, input_matrix.shape[1] - pool_size + 1, stride):
            region = input_matrix[i:i + pool_size, j:j + pool_size]
            if mode == 'max':
                pooled_matrix[i // stride, j // stride] = np.max(region)
            elif mode == 'avg':
                pooled_matrix[i // stride, j // stride] = np.mean(region)
    return pooled_matrix


def plot_heatmap(
    ax, title, data, vmin, vmax, fmt,
    cmap='cividis', annot=True, cbar=False, square=True,
):
    ax.axis('off')
    ax.set_title(title)
    sns.heatmap(data, vmin=vmin, vmax=vmax, annot=True, fmt=fmt, linewidth=.5, cmap=cmap, cbar=False, square=True, ax=ax)


def plot_matrixes(input_matrix, output_matrix, vmin, vmax, fmt, title='Convolution (Matrix)'):
    sns.set(style='whitegrid', context='talk')

    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [7, 3]})
    fig.suptitle(title)

    plot_heatmap(axes[0], 'Input', input_matrix, vmin, vmax, fmt)
    plot_heatmap(axes[1], 'Output', output_matrix, vmin, vmax, fmt)

    fig.text(1.15, 0.5, '->', fontsize=20, va='center', ha='center', transform = axes[0].transAxes, color ='black')

    plt.tight_layout()
    plt.show()


def sample_max_pooling():
    input_matrix = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [7, 8, 3, 0],
        [2, 1, 4, 5],
    ])
    output_matrix = pooling(input_matrix, pool_size=2, stride=2, mode='max')
    plot_matrixes(input_matrix, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Pooling (max)')


def sample_avg_pooling():
    input_matrix = np.array([
        [1, 3, 2, 4],
        [5, 6, 1, 2],
        [7, 8, 3, 0],
        [2, 1, 4, 5],
    ])
    output_matrix = pooling(input_matrix, pool_size=2, stride=2, mode='avg')
    plot_matrixes(input_matrix, output_matrix, vmin=-30, vmax=30, fmt='.0f', title='Pooling (avg)')


if __name__ == "__main__":
    #sample_max_pooling()
    sample_avg_pooling()
