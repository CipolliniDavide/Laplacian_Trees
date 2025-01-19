#! /usr/bin/env python3

import os
import sys

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy.linalg import eigh

# from helpers import visualize
# from helpers import utils
# from helpers.utils import unroll_nested_list, ensure_dir
from helpers.visual_utils import set_ticks_label, set_legend, create_colorbar, get_set_larger_ticks_and_labels, \
    create_n_colors_from_cmap


def plot_eigen(curve_labels: np.array,
               eigen_list_of_list: list = None,
               graph_list_gt: nx.Graph = None,
               index_to_plot: list = None,
               save_path: str = None,
               figname: str = None,
               cmap: str = 'viridis',
               eigenwhat: str='eigenvalues',
               figsize: tuple = (6, 4.5),
               legend_ncol: int = 2,
               fig_format: str = 'png',
               title: str = None,
               show: bool = False,
               reference_curve: bool=True):

    if (eigen_list_of_list is None):
        eigen_list_of_list = list()
        for gt in graph_list_gt:
            eigen_list_of_list.append(eigh(nx.laplacian_matrix(gt).toarray(), eigvals_only=True))

    n_colors = len(curve_labels) + 2
    cmap_name = cmap
    colors = create_n_colors_from_cmap(cmap_name, n_colors)

    if index_to_plot is None:
        index_to_plot = list(range(len(curve_labels)))

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize, layout='tight')
    # Compute PDF using kernel density estimation (KDE)
    for i, ind in enumerate(index_to_plot):
        eigenvalues = np.sort(eigen_list_of_list[ind])[1:]
        ax.plot(2 + np.arange(len(eigenvalues)), eigenvalues, color=colors[::-1][ind],
                alpha=.3, linewidth=3,
                label='{:.2f}'.format(curve_labels[ind]), marker='.')

    if reference_curve:
        g = nx.grid_graph(dim=(len(eigenvalues)+1, 1), periodic=True)
        g.remove_edges_from(nx.selfloop_edges(g))
        L = nx.laplacian_matrix(g).toarray()
        lambd, Q = np.linalg.eigh(L)  # eigenvalues and eigenvectors of (batched) Laplacian
        ax.loglog(range(2, g.number_of_nodes()+1), np.sort(lambd)[1:], linestyle='--', color='red', label='1d lattice',
                  linewidth=2, alpha=.8)

    ax.set_yscale('log')
    ax.set_xscale('log')
    get_set_larger_ticks_and_labels(ax=ax)
    if eigenwhat == 'eigenvalues':
        ax.set_ylabel(r'$\mathbf{\lambda_i}$')
    else:
        ax.set_ylabel(r'$\mathbf{\bar{|\lambda_i>}}$')
    ax.set_xlabel(r'$\mathbf{i}$')
    ax.set_xlim(left=1)
    # Set font properties for the y-axis label
    ax.tick_params(axis='both', which='both', width=2, length=7.5)
    ax.tick_params(axis='both', which='minor', width=2, length=4)
    set_legend(ax, title='r', ncol=legend_ncol, loc=0)
    ax.tick_params(axis='y', which='minor', bottom=False)
    if title:
        ax.set_title(title)
    # plt.grid()

    if save_path is not None:
        plt.savefig(save_path + figname + f'.{fig_format}', dpi=300)
    if show:
        plt.show()
    else:
        plt.close


def plot_propagator_eigenvalues(tau_range: np.array, lrho: np.array, save_dir: str = '{:s}'.format(os.getcwd()),
                           title: str = '', fig_format: str = 'png', figsize: tuple = (7, 10),
                           tau_stars: np.array=None,
                           tau_lim: tuple=(5, 30),
                           cmap_name: str = 'viridis',
                                save_name: str='propagator_r'
                           ):

    fig = plt.figure(figsize=figsize, layout='tight')
    plt.suptitle(title)
    ax1 = plt.subplot(2, 1, 1)
    x, y = np.meshgrid(np.arange(len(lrho[0])), np.log10(tau_range))
    map = ax1.pcolormesh(x, y, lrho, cmap=cmap_name)
    # ax1.set_xticks([])
    set_ticks_label(ax=ax1, ax_type='y',
                    # data=np.log10(tau_range),
                    data=np.log10(tau_range),
                    num=5,
                    valfmt="{x:.0f}",
                    ticks=None,
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=r'$\mathbf{\log_{10}{\tau}}$',
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    add_ticks=[])

    set_ticks_label(ax=ax1, ax_type='x',
                    data=np.arange(len(lrho[0])),
                    num=5,
                    valfmt="{x:.0f}",
                    ticks=None,
                    only_ticks=False, tick_lab=None,
                    fontdict_ticks_label={'weight': 'bold', 'size': 'x-large'}, label_pad=4,
                    ax_label=r'$\mathbf{i}$',
                    fontdict_label={'weight': 'bold', 'size': 'xx-large', 'color': 'black'}, scale=None,
                    add_ticks=[])

    get_set_larger_ticks_and_labels(ax=ax1, num_ticks_x=6)

    create_colorbar(fig=fig, ax=ax1, mapp=map,
                    array_of_values=lrho,
                    valfmt="{x:.2f}",
                    fontdict_cbar_label={'label': r'$\mathbf{\lambda_i}$'},
                    fontdict_cbar_tickslabel=None, fontdict_cbar_ticks=None, position='right')

    # lrho_downsample = lrho[:, ::200]
    cmap = cm.get_cmap('viridis')
    num_curves = len(lrho[1])
    norm = mcolors.LogNorm(vmin=1, vmax=num_curves)

    # color_indices = np.logspace(0, np.log10(num_curves - 1), num_curves, base=10, dtype=int)
    color_indices = range(num_curves)
    colors = [cmap(norm(i+1)) for i in color_indices]
    ax2 = plt.subplot(2, 1, 2)
    for i in np.linspace(start=0, stop=lrho.shape[1]-1, endpoint=True, num=2000, dtype=int):
        ax2.semilogx(tau_range, lrho[:, i], color=colors[i], label='{:d}'.format(i), alpha=.3, linewidth=4)
    # get_set_larger_ticks_and_labels(ax=ax2, num_ticks_x=6)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[::200], labels[::200])
    # set_legend(ax=ax2)
    plt.savefig('{:s}/{:s}.{:s}'.format(save_dir, save_name, fig_format), dpi=300)
    plt.show()
    plt.close()
    a=0


def plot_propagator(lrho: np.array):
    from mpl_toolkits.mplot3d import Axes3D

    # Downsample factor
    downsample_factor = 2

    # Downsample lrho
    lrho_downsampled = lrho[::downsample_factor, ::downsample_factor, :]

    # Create meshgrid for x, y, and z coordinates
    x, y, z = np.meshgrid(np.arange(0, lrho.shape[0], downsample_factor),
                          np.arange(0, lrho.shape[1], downsample_factor),
                          np.arange(0, lrho.shape[2], 1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot a surface
    surf = ax.scatter(x, y, z, c=lrho_downsampled.flatten(), cmap=plt.cm.viridis)

    # Set labels
    ax.set_xlabel(r'$\mathbf{\tau}$')
    ax.set_ylabel('r')
    ax.set_zlabel('i')

    # Show color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    a=0