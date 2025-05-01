import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def create_grid_figure(widths, figsize, wspace=0.1):
    """
    Create a figure with a GridSpec layout and transparent background.

    Parameters
    ----------
    widths : tuple of int
        Relative widths of the subplots.
    figsize : tuple of float
        Figure size (width, height).
    wspace : float
        Width space between subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
    gs : matplotlib.gridspec.GridSpec
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('none')
    gs = gridspec.GridSpec(nrows=1, ncols=len(widths), width_ratios=widths)
    gs.update(wspace=wspace)
    return fig, gs


def add_segmentation_lines(ax, max_trial, interval=64, **line_kwargs):
    """
    Add vertical segmentation lines every `interval` trials.
    """
    for x in range(interval, max_trial + 1, interval):
        ax.axvline(x=x, **line_kwargs)


def style_axis(ax, show_ylabel=False, yticks=(0, 0.5, 1.0),
               yticklabels=('0', '0.5', '1'), xlabel='Trial',
               ylabel='Accuracy', xtick_interval=64, fontsize=18, tick_fontsize=15):
    """
    Apply common styling to an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    show_ylabel : bool
        Whether to display y-axis ticks and label.
    yticks : tuple
        Y-axis tick positions.
    yticklabels : tuple
        Y-axis tick labels.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    fontsize : int
        Font size for axis labels.
    tick_fontsize : int
        Font size for tick labels.
    """
    ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, xtick_interval))
    ax.set_xticklabels(range(0, int(ax.get_xlim()[1]) + 1, xtick_interval), fontsize=tick_fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)

    if show_ylabel:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=tick_fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
    else:
        ax.set_yticks([])
        ax.set_ylabel('')

    ax.grid(False)
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2)
    ax.tick_params(width=2)

def annotate_label(ax, text, loc=(0.95, 0.05), fontsize=18,
                   ha='right', va='bottom', color='black'):
    """
    Annotate a subplot with a label at a relative location.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    text : str
        Text label, e.g., subject ID.
    loc : tuple
        Relative (x, y) location in axes coordinates.
    fontsize : int
    ha : str
        Horizontal alignment.
    va : str
        Vertical alignment.
    color : str
    """
    ax.text(loc[0], loc[1], text,
            transform=ax.transAxes,
            fontsize=fontsize,
            ha=ha,
            va=va,
            color=color)
