import os
import matplotlib.cm as cm

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import glob
import igraph as ig
import pickle
import pandas as pd
from scipy.linalg import expm
from scipy.sparse import csc_matrix, csr_array
from scipy.sparse.linalg import expm as sparse_expm
from helpers.visual_utils import set_ticks_label, get_set_larger_ticks_and_labels, set_legend
from helpers_dataset.helpers_dataset import (order_files_by_r,
                                             convert_tree_to_nx)

import sys
print(sys.getrecursionlimit())

def plot_(x, y, yerr, n_nodes, fig_format='pdf', ylabel='Modularity', ylabel_var="$\mathbf{2m\cdot Var(Modularity)}$",
          x_ticks=[0, .4, .9, 1, 1.5, 2], x_lim=(0, .95), name='modularity'):
    # fig, axes = plt.subplots(figsize=(9, 4), ncols=1, tight_layout=True)
    fig, ax1 = plt.subplots(figsize=(6, 4), ncols=1, tight_layout=True)
    ax2 = ax1.twinx()

    for i in range(len(y)):
        if len(y[i]) > len(x):
            y[i] = y[i][1:]
        if len(yerr[i]) > len(x):
            yerr[i] = yerr[i][1:]
        # axes[0].plot(r_, avg_modularity_list[i], label=f'{n_nodes[i]:d}', c=cmap(i))
        # axes[1].plot(r_, var_modularity_list[i] * 2 * (n_nodes[i] - 1), label=f'{n_nodes[i]:d}', c=cmap(i))
        ax1.plot(x, y[i], label=f'{n_nodes[i]:d}', c=cmap(i))
        ax2.plot(x, yerr[i], label=f'{n_nodes[i]:d}', c=cmap(i), linestyle='--')

    axes = [ax1, ax2]
    # [ax.set_xlabel('r') for ax in axes_mod]
    axes[0].set_ylabel(ylabel)
    axes[1].set_ylabel(ylabel_var)
    axes[0].set_ylim([0.00, 1.01])
    axes[1].set_ylim(bottom=0.00)
    [set_ticks_label(ax=ax, ax_type='x', ax_label='r', data=[0, 2], num=5, valfmt="{x:.1f}",
                     ticks=x_ticks)
     for ax in axes]
    [ax.set_xlim(x_lim) for ax in axes]
    [get_set_larger_ticks_and_labels(ax=ax) for ax in axes]
    [ax.grid(axis='x') for ax in axes]
    [set_legend(ax[0], loc=2) for ax in [axes]]
    save_name_mod = save_fold + name

    fig.savefig(save_name_mod + f".{fig_format}", dpi=300)
    plt.show()


def save_array_modularity(dict_files, vert_gap):
    print(load_ntw_dir)
    hier = [[] for _ in range(len(r_))]
    modularity = [[] for _ in range(len(r_))]

    for i, key in enumerate(r_):
        print('r=', key, len(dict_files[key]))
        for file_name in dict_files[key]:
            G = convert_tree_to_nx(file_name)
            # prop_file = file_name.replace('.tree', '.pkl').replace('output', 'prop')
            # diameter = pickle.load(open(prop_file, 'rb'))['diameter']
            # hier[i].append(diameter)
            if G.number_of_nodes() == 0:
                print('Error: ', file_name)
                # modularity[i].append(np.nan)
            else:
                # Compute modularity
                h = ig.Graph.from_networkx(G)
                communities = h.community_multilevel()
                # if key > .4:
                #     print(f"r={key}: num comunities {len(communities)}")
                Q = h.modularity(communities)
                modularity[i].append(Q)

                try:
                    # # Hierarchy
                    if is_chain_1d(G):
                        root_node = list(G.nodes(data=True))[0][0]
                    else:
                        root_node = list(G.nodes(data=True))[1][0]
                    pos = - np.array([p[1] for p in hierarchy_pos(G, root=root_node, vert_gap=vert_gap).values()]) / vert_gap
                    hier[i].append(pos.max())
                except:
                    pass
    # for r, h in zip(r_, modularity):
    #     print(r, np.mean(h), '+-', np.var(h))

    avg_modularity_list[ind] = np.array([np.mean(h) for h in modularity])
    var_modularity_list[ind] = np.array([np.var(h) for h in modularity])
    np.save(file=f'{save_fold}/avg_modularity_fashionmnist_nodes{nodes}.npy', arr=avg_modularity_list[ind])
    np.save(file=f'{save_fold}/var_modularity_fashionmnist_nodes{nodes}.npy', arr=var_modularity_list[ind])

    avg_hierarchy_list[ind] = np.array([np.mean(h) for h in hier])
    var_hierarchy_list[ind] = np.array([np.var(h) for h in hier])
    np.save(file=f'{save_fold}/avg_hierarchy_fashionmnist_nodes{nodes}.npy', arr=avg_hierarchy_list[ind])
    np.save(file=f'{save_fold}/var_hierarchy_fashionmnist_nodes{nodes}.npy', arr=var_hierarchy_list[ind])


def is_chain_1d(graph):
    # A graph is a chain if it is connected and has exactly two nodes with degree 1,
    # and all other nodes have degree 2.
    if not nx.is_connected(graph):
        return False

    degree_sequence = [d for n, d in graph.degree()]

    # Count the nodes with degree 1 and 2
    count_deg_1 = degree_sequence.count(1)
    count_deg_2 = degree_sequence.count(2)

    # There should be exactly two nodes with degree 1 (the endpoints of the chain),
    # and all other nodes should have degree 2.
    if count_deg_1 == 2 and count_deg_2 == (len(degree_sequence) - 2):
        return True

    return False


def plot_hierarchical_tree(G, root=None, node_size=30, with_labels=False, node_color='blue',
                           fig_size=(10, 8),
                           cmap=plt.cm.tab10, arrows=False, ax=None):
    pos = hierarchy_pos(G, root)
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color=node_color, font_size=10, font_weight="bold",
            arrows=arrows, ax=ax, cmap=cmap)
    # plt.show()


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos


def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)

    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                 pos=pos, parent=root, parsed=parsed)

    return pos


if __name__ == '__main__':
    fig_format = 'png'
    load_graph = nx.read_graphml
    save_fold = 'figures_modularity/'
    os.makedirs(save_fold, exist_ok=True)

    vert_gap = 0.2

    dir_list = [
                 'boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=500,500-s=1',
                'boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=1000,1000-s=1',
                # 'boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019',
                # 'boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1'
                ]
    cmap_ = 'plasma'
    cmap = cm.get_cmap(cmap_, len(dir_list) + 1)

    avg_modularity_list = [[] for _ in range(len(dir_list))]
    var_modularity_list = [[] for _ in range(len(dir_list))]

    avg_hierarchy_list = [[] for _ in range(len(dir_list))]
    var_hierarchy_list = [[] for _ in range(len(dir_list))]

    n_nodes = [[] for _ in range(len(dir_list))]

    for ind, load_ntw_dir in enumerate(dir_list):

        dict_files = order_files_by_r(file_list=sorted(glob.glob("{:s}/Dataset/{:s}/output/*.tree".format(os.getcwd(), load_ntw_dir))))
        map_files = order_files_by_r(file_list=sorted(glob.glob("{:s}/Dataset/{:s}/output/*.map".format(os.getcwd(), load_ntw_dir))))

        r_ = np.array(list(dict_files.keys()))

        clustering_list = [[] for _ in range(len(r_))]
        file_name = dict_files[r_[1]][0]
        G = convert_tree_to_nx(file_name)
        nodes = G.number_of_nodes()
        n_nodes[ind] = nodes

        # Create the data
        # save_array_modularity(dict_files=dict_files, vert_gap=vert_gap)

        avg_modularity_list[ind] = np.load(file=f'{save_fold}/avg_modularity_fashionmnist_nodes{nodes}.npy')
        var_modularity_list[ind] = np.load(file=f'{save_fold}/var_modularity_fashionmnist_nodes{nodes}.npy')

        avg_hierarchy_list[ind] = np.load(file=f'{save_fold}/avg_hierarchy_fashionmnist_nodes{nodes}.npy')
        var_hierarchy_list[ind] = np.load(file=f'{save_fold}/var_hierarchy_fashionmnist_nodes{nodes}.npy')

    plot_(x=r_,
          y=avg_modularity_list,
          yerr=[var_modularity_list[i] * 2*(n_nodes[i]-1) for i in range(len(n_nodes))],
          n_nodes=n_nodes,
          fig_format=fig_format,
          ylabel='Modularity', ylabel_var="$\mathbf{2m\cdot Var(Modularity)}$",
          x_ticks=[0, .4, .9, 1, 1.5, 2], x_lim=(0, .95), name='modularity')

    plot_(x=r_,
          y=[avg_hierarchy_list[i]/(n_nodes[i] - 1) for i in range(len(n_nodes))],
          yerr=var_hierarchy_list,
          n_nodes=n_nodes,
          fig_format=fig_format,
          ylabel='$\mathbf{\hat{h}}$',
          ylabel_var='$\mathbf{(N-1) \cdot Var(\hat{h})}$',
          x_ticks=[0, .5, 1, 1.5, 2],
          x_lim=(0, 2), name='hierarchy')

    #########################################################################################################
    # Diameter
    avg_diameter_list = [[] for _ in range(len(dir_list))]
    var_diameter_list = [[] for _ in range(len(dir_list))]
    n_nodes = [[] for _ in range(len(dir_list))]
    for ind, load_ntw_dir in enumerate(dir_list):
        dict_files = order_files_by_r(
            file_list=sorted(glob.glob("{:s}/Dataset/{:s}/output/*.tree".format(os.getcwd(), load_ntw_dir))))

        print(load_ntw_dir)
        hier = [[] for _ in range(len(r_))]
        for i, key in enumerate(r_):
            # print('r=', key, len(dict_files[key]))
            for file_name in dict_files[key]:
                prop_file = file_name.replace('.tree', '.pkl').replace('output', 'prop')
                diameter = pickle.load(open(prop_file, 'rb'))['diameter']
                hier[i].append(diameter)
        nodes = pickle.load(open(prop_file, 'rb'))['n_nodes']
        print(nodes)
        n_nodes[ind] = nodes
        avg_diameter_list[ind] = np.array([np.mean(h) for h in hier])
        var_diameter_list[ind] = np.array([np.var(h) for h in hier])


    plot_(x=r_,
          y=[avg_diameter_list[i]/(n_nodes[i]-1) for i in range(len(n_nodes))],
          yerr=var_diameter_list,
          n_nodes=n_nodes,
          fig_format=fig_format,
          ylabel='$\mathbf{\hat{d}}$',
          ylabel_var='(N-1)' + '$\mathbf{\cdot}$' + '$\mathbf{Var(\hat{d})}$',
          x_ticks=[0, .5, 1, 1.5, 2], x_lim=(0, 2), name='diameter')

