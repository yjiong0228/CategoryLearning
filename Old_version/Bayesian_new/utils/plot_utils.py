# plot_utils.py
import matplotlib.pyplot as plt
import numpy as np

COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def init_figure(n_rows, n_cols, figsize=(25, 5), hspace=0.5, wspace=0.5):
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(figsize[0], figsize[1] * n_rows),
                            facecolor='none',
                            squeeze=False)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    return fig, axs

def style_subject_ax(ax, *,
                     row, col, n_rows, n_cols,
                     num_steps, condition,
                     subject_idx,
                     y_range=None,
                     n_yticks=6,
                     y_tick_format='decimal'):
    """
    对单个子图应用统一样式，并可控制 y 轴范围、刻度数量与标签格式。

    参数:
      row, col: 子图的行列索引
      n_rows, n_cols: 总行数和列数
      num_steps: x 轴最大步数
      condition: 条件编号，用于选择不同的 x 步长
      subject_idx: 被试编号，用于右下角标注
      y_range: (ymin, ymax)，如果不为 None，则设置 y 轴范围
      n_yticks: y 轴刻度总数（包括端点）
      y_tick_format: 'int' 或 'decimal'，控制 y 轴标签显示整数还是保留一位小数
    """
    # 绘制竖直分段线
    for x in range(64, num_steps + 1, 64):
        ax.axvline(x=x, color='grey', alpha=0.3,
                   linestyle='dashed', linewidth=1)

    # 设置 y 轴范围与刻度
    if y_range is not None:
        ymin, ymax = y_range
        ax.set_ylim(ymin, ymax)
        yticks = np.linspace(ymin, ymax, n_yticks)
        ax.set_yticks(yticks)
        # 根据格式选择标签
        if y_tick_format == 'int':
            labels = [f"{int(round(y))}" for y in yticks]
        else:
            labels = [f"{y:.1f}" for y in yticks]
        ax.set_yticklabels(labels, fontsize=15)
    else:
        # 仅第一列显示默认刻度
        if col == 0:
            yticks = [i / 5 for i in range(6)]
            ax.set_yticks(yticks)
            if y_tick_format == 'int':
                labels = [f"{int(round(y))}" for y in yticks]
            else:
                labels = [f"{y:.1f}" for y in yticks]
            ax.set_yticklabels(labels, fontsize=15)
        else:
            ax.set_yticks([])
            ax.set_ylabel(None)

    # 设置 x 轴刻度，仅最后一行显示
    if row == n_rows - 1:
        step = 64
        ax.set_xticks(range(0, num_steps + 1, step))
        ax.set_xticklabels(range(0, num_steps + 1, step), fontsize=15)
    else:
        ax.set_xticks([])
        ax.set_xlabel(None)

    # 去除网格和多余 spine
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.tick_params(width=2.0)
    ax.set_facecolor('none')

    # 右下角标注被试号
    # ax.text(0.95, 0.05, f"S{subject_idx}",
    #         transform=ax.transAxes,
    #         fontsize=30, ha='right', va='bottom',
    #         color='black')

    return ax

def add_global_labels(fig, *,
                      xlabel="Trial", ylabel="Accuracy",
                      xlabel_fontsize=25, ylabel_fontsize=25):
    """给整张 figure 增加横纵坐标标题"""
    fig.text(0.5, 0.02, xlabel,
             ha='center', fontsize=xlabel_fontsize)
    fig.text(0.01, 0.5, ylabel,
             va='center', rotation='vertical',
             fontsize=ylabel_fontsize)
    return fig


