import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

colors = ['red', 'green', 'blue', 'orange', 'black', 'cyan', 'magenta', 'pink', 'yellow', 'gray', 'lightblue',
          'lightgreen', 'purple', 'brown', 'teal', 'olive', 'navy', 'lime', 'coral', 'salmon', 'aqua', 'wheat', 'white']
styles = ['-', '--', '-.', ':']
legend_loc = 'best'
display_threshold = 1
fig_width = 433.62
n_ticks = 5


def plot_q(ts, qs, q_des, save_path, ylim=None, ax=None, set_legend=True, figure=None):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    n_state = q_des.shape[-1]

    for i in range(n_state):
        if i >= len(colors):
            continue
        for j, q in enumerate(qs):
            ax.plot(ts[:], q[:, i], linestyle=styles[j], color=colors[i],
                    label=f'$q_{i + 1}(t)$')
        ax.plot(ts, q_des[:, i], label=f'$q_{{des,{i + 1}}}(t)$', linestyle=':',
                color=colors[i])
    if ylim is not None:
        try:
            ax.set_ylim(ylim)
        except:
            ...
    if set_legend:
        # ax.set_xlabel('Time t')
        if n_state < display_threshold:
            ax.legend(loc=legend_loc)
        else:
            ax.legend(handles=[Line2D([0], [0], color='black', linestyle='-'),
                               Line2D([0], [0], color='black', linestyle=':')],
                      labels=[f'$q(t)$', f'$q_{{des}}(t)$'], loc=legend_loc)
    if figure is not None and save_path is not None:
        figure.savefig(save_path)
        figure.clear()
        plt.close(figure)


def plot_control(ts, U, save_path, n_point_delay, ylim=None, ax=None, set_legend=True, figure=None, linestyle='-'):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

    assert U.ndim == 2
    U = U.T
    for i, u in enumerate(U):
        if i >= len(colors):
            continue
        ax.plot(ts[n_point_delay:], u[n_point_delay:], label=f'$U_{i + 1}(t)$', color=colors[i], linestyle=linestyle)
    if ylim is not None:
        try:
            ax.set_ylim(ylim)
        except:
            ...
    if set_legend:
        ax.legend(handles=[Line2D([0], [0], color='black', linestyle='-')],
                  labels=[f'$U(t)$'], loc=legend_loc)

    if figure is not None and save_path is not None:
        figure.savefig(save_path)
        figure.clear()
        plt.close(figure)


def plot_result(dataset_config, img_save_path, P_no, P_numerical, Z, U, method):
    if img_save_path is None:
        return
    ts = dataset_config.ts
    delay = dataset_config.delay
    n_point_delay = dataset_config.n_point_delay
    comparison_full = f'{img_save_path}/{method}_comp_fit.png'
    difference_full = f'{img_save_path}/{method}_diff_fit.png'
    comparison_zoom = f'{img_save_path}/{method}_comp.png'
    difference_zoom = f'{img_save_path}/{method}_diff.png'
    u_path = f'{img_save_path}/{method}_u.png'
    if method == 'no':
        plot_comparison(ts, [P_no], Z, delay, n_point_delay, comparison_full)
        plot_difference(ts, [P_no], Z, n_point_delay, difference_full)
        plot_comparison(ts, [P_no], Z, delay, n_point_delay, comparison_zoom, ylim=[-5, 5])
        plot_difference(ts, [P_no], Z, n_point_delay, difference_zoom, ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    elif method == 'numerical':
        plot_comparison(ts, [P_numerical], Z, delay, n_point_delay, comparison_full)
        plot_difference(ts, [P_numerical], Z, n_point_delay, difference_full)
        plot_comparison(ts, [P_numerical], Z, delay, n_point_delay, comparison_zoom, ylim=[-5, 5])
        plot_difference(ts, [P_numerical], Z, n_point_delay, difference_zoom, ylim=[-5, 5])
        plot_control(ts, U, u_path, n_point_delay)
    else:
        raise NotImplementedError()


def plot_comparison(ts, Ps, Z, delay, n_point_delay, save_path, ylim=None, Ps_labels=None, ax=None, set_legend=True,
                    figure=None):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

    n_state = Z.shape[-1]
    if Ps_labels is None:
        Ps_labels = ['' for _ in range(len(Ps))]

    linesytles = ['--', ':']
    for i in range(n_state):
        if i >= len(colors):
            continue
        for j, (P, label) in enumerate(zip(Ps, Ps_labels)):
            ax.plot(ts[2 * n_point_delay:], P[n_point_delay:-n_point_delay, i], linestyle=linesytles[j],
                    color=colors[i], label=f'$P^{{{label}}}_{i + 1}(t-{delay})$')
        ax.plot(ts[n_point_delay:], Z[n_point_delay:, i], label=f'$Z_{i + 1}(t)$', linestyle='-', color=colors[i])
    if ylim is not None:
        try:
            ax.set_ylim(ylim)
        except:
            ...

    if set_legend:
        if n_state < display_threshold:
            ax.legend(loc=legend_loc)
        else:
            handles = [
                Line2D([0], [0], color='black', linestyle='--'),
                Line2D([0], [0], color='black', linestyle='-')
            ]
            labels = [f'$P(t-{delay})$', f'$Z(t)$']
            if len(Ps) == 2:
                handles.append(Line2D([0], [0], color='black', linestyle='-'))
                labels.append(rf'$P^\prime(t)$')
            ax.legend(handles=handles,
                      labels=labels, loc=legend_loc)
    if figure is not None and save_path is not None:
        figure.savefig(save_path)
        figure.clear()
        plt.close(figure)


def difference(Z, P, n_point_delay):
    return P[:-n_point_delay] - Z[n_point_delay:]


def plot_difference(ts, Ps, Z, n_point_delay, save_path, ylim=None, Ps_labels=None, ax=None, set_legend=True,
                    differences=None, figure=None, xlim=None):
    if ax is None:
        figure = plt.figure(figsize=set_size(width=fig_width))
        ax = figure.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    n_state = Z.shape[-1]

    if Ps_labels is None:
        Ps_labels = ['' for _ in range(len(Ps))]
    if differences is None:
        differences = []
        for P in Ps:
            differences.append(difference(Z, P, n_point_delay))

    for i in range(n_state):
        if i >= len(colors):
            continue
        for j, (d, label) in enumerate(zip(differences, Ps_labels)):
            ts_ = ts[n_point_delay:-n_point_delay] if n_point_delay != 0 else ts[n_point_delay:]
            ax.plot(ts_, abs(d[n_point_delay:, i]), linestyle=styles[j], color=colors[i],
                    label=f'$\Delta P^{{{label}}}_{i + 1}(t)$')

    if ylim is not None:
        try:
            ax.set_yscale('log')
            # y_locator = LogLocator(base=10.0, numticks=5)
            # ax.yaxis.set_major_locator(y_locator)
            # ax.yaxis.set_major_formatter(LogFormatterMathtext())
            # ax.set_ylim([0, ylim[1]])
            ax.set_yticks([1e-6, 1e-4, 1e-2, 1])
            # ax.set_ylim([0, 100])
        except:
            ...
    if xlim is not None:
        ax.set_xlim(xlim)
    if set_legend:
        # ax.set_xlabel('Time t')
        if n_state < display_threshold:
            ax.legend(loc=legend_loc)
        else:
            ax.legend(handles=[Line2D([0], [0], color='black', linestyle='-')],
                      labels=[f'$\Delta P(t)$'], loc=legend_loc)
    if figure is not None and save_path is not None:
        figure.savefig(save_path)
        figure.clear()
        plt.close(figure)


def set_size(width=None, fraction=1, subplots=(1, 1), height_add=0.1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif width is None:
        width_pt = fig_width
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = height_add + fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in
