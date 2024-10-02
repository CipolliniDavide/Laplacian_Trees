from matplotlib import pyplot as plt
import numpy as np
from helpers.utils import unroll_nested_list, ensure_dir
from helpers.visual_utils import set_ticks_label, set_legend, create_colorbar, get_set_larger_ticks_and_labels
from .theor_deg_distr_rand_graph import plot_degree_distribution, rand_degree_distribution

def compare_bar_plot(deg_list_of_list, save_fold=None,
                     binning_type='log', save_name='mean_degree_std.png',
                     colors=['blue', 'red'], labels=['BFO', 'Voronoi'],
                     figsize=(6, 5), labelx='k', labely='P(k)', show=False,
                     fig_format='.pdf', legend_title=None,
                     loglog=True,
                     name_fig='histogram_Pk',
                     show_erdos_renyi=False):

    fig, ax = plt.subplots(figsize=figsize)
    deg_un_list = []
    c_l = []

    for i, deg in enumerate(deg_list_of_list):
        # flat_list = [item for sublist in deg for item in sublist]
        if binning_type == 'linear':
            deg_unique, counts = np.unique(deg, return_counts=True)
            deg_un_list.append(deg_unique)
            c = counts / len(deg)
            c_l.append(c)
            ax.scatter(deg_unique, c, linewidth=3, label=f'iter{i}') #, color=colors[i])

        elif binning_type == 'log':
            max_degree = np.max(deg)
            num_bins = int(np.ceil(np.log2(max_degree)))
            bins = np.logspace(0, num_bins, num=num_bins + 1, base=2, dtype=int)
            counts, _ = np.histogram(deg, bins)
            c = counts / len(deg)
            c_l.append(c)
            ax.scatter(bins[:-1], c, linewidth=3, label=f'iter{i}', marker='o')
            deg_un_list.append([1, 1e4])
        else:
            raise ValueError(f'Binning type not specified. Options are linear, log')

    # ax.set_title('All 16 crops together')
    deg_un_list = np.unique(unroll_nested_list(deg_un_list))
    set_ticks_label(ax=ax, ax_type='x', data=deg_un_list, valfmt="{x:.0f}",
                    ax_label=labelx)
    # set_ticks_label(ax=ax, ax_type='y', data=utils.unroll_nested_list(c_l), valfmt="{x:.1f}", ax_label=labely)
    font_properties = {'weight': 'bold', 'size': 'xx-large'}
    if loglog:
        ax.set_yscale('log')
        ax.set_xscale('log')
    ax.set_ylabel(labely)
    y_label = ax.yaxis.get_label()
    y_label.set_font_properties(font_properties)
    y_tick_labels = ax.get_yticklabels()
    for label in y_tick_labels:
        label.set_font_properties({'weight': 'bold', 'size': 'xx-large'})
    ax.set_ylim(top=1)
    ax.set_xlim(right=1e4)
    plt.grid(linewidth=.5)

    if show_erdos_renyi:
        degrees, probabilities = rand_degree_distribution(n=deg.shape[0], m=deg.shape[0]-1)
        ax.plot(degrees, probabilities, linestyle='--', linewidth=2, label=f'Erdos-Renyi')

    set_legend(ax=ax, title=f'r={legend_title}')

    plt.tight_layout()


    if save_fold:
        plt.savefig(save_fold + '{:s}_{:s}.{:s}'.format(name_fig, legend_title, fig_format))
    if show:
        plt.show()
    else:
        plt.close()
