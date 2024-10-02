
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os
from scipy.linalg import expm
from scipy.sparse import csc_matrix, csr_array
from scipy.sparse.linalg import expm as sparse_expm


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

from helpers import utils
from helpers_dataset.helpers_dataset import (order_files_by_r,
                                             convert_tree_to_nx)


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
    fig_format = 'pdf'
    load_ntw_dir = f"{os.getcwd()}/Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=20,20-s=1/output/"

    # load_ntw_dir = f"{os.getcwd()}/Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=50,50-s=1/output/"
    # load_ntw_dir = f"{os.getcwd()}/Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=100,100-s=1/output/"
    # load_ntw_dir = f"{os.getcwd()}/Dataset/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019/output/'
    #
    save_dir = load_ntw_dir.replace('output/', '')
    load_graph = nx.read_graphml
    dict_files = order_files_by_r(file_list=sorted(glob.glob("{:s}/*.tree".format(load_ntw_dir))))
    map_files = order_files_by_r(file_list=sorted(glob.glob("{:s}/*.map".format(load_ntw_dir))))

    r_ = np.array(list(dict_files.keys()))

    r_to_plot = [.05, .95, r_[-1]]
    # r_to_plot = [.35, .7, r_[-1]]

    name_fig = '{:.2f}_{:.2f}_{:.2f}'.format(r_to_plot[0], r_to_plot[1], r_to_plot[2])

    fig, axes = plt.subplots(figsize=(6*len(r_to_plot), 6), ncols=len(r_to_plot), layout='tight')

    for i, (key, ax) in enumerate(zip(r_to_plot, axes)):
        file_name = dict_files[key][1]
        print(file_name)
        # file_name_prop = file_name.replace('output', 'prop').replace('.tree', '')
        # prop = utils.pickle_load(file_name_prop)

        class_mapping_df = pd.read_csv(map_files[key][0], sep=":", header=None, skiprows=1, names=["Index", "Class"])
        class_mapping = dict(class_mapping_df.values)

        G = convert_tree_to_nx(file_name)
        node_classes = [class_mapping[int(str(node).replace('N', ''))] for node in G.nodes()]

        # Plot del grafo con colori basati sulle classi
        pos = nx.spring_layout(G, seed=1)  # posizionamento dei nodi
        nx.draw(G, pos, with_labels=False, node_color=node_classes, cmap=plt.cm.tab10, node_size=300, ax=ax)

        # if is_chain_1d(G):
        #     root_node = list(G.nodes(data=True))[0][0]
        # else:
        #     root_node = list(G.nodes(data=True))[1][0]
        #
        # plot_hierarchical_tree(G,
        #                        root=root_node,
        #                        ax=axes[1],
        #                        fig_size=(12, 8),
        #                        node_size=100, with_labels=False,
        #                        node_color=node_classes,
        #                        arrows=False, cmap=plt.cm.tab10)
        # plt.show()

    sm = plt.cm.ScalarMappable(cmap=plt.cm.tab10,
                               norm=plt.Normalize(vmin=min(node_classes), vmax=max(node_classes)))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, ticks=range(min(node_classes), max(node_classes) + 1), label='Class')
    cbar.ax.tick_params(labelsize=20)  # Imposta la dimensione del carattere delle tick labels
    cbar.ax.set_ylabel('Class', fontsize=24, fontweight='bold')

    plt.savefig(save_dir+'network_trasformation_r{:s}_nodes{:d}.{:s}'.format(name_fig, G.number_of_nodes(), fig_format), dpi=300)
    plt.show()



    ################################# Hierarchical plot ##############################
    n_rows = 7
    r_to_plot = np.round(np.arange(0.0, 1.5, .05), decimals=2)[::2]
    fig, axes = plt.subplots(figsize=(6 * len(r_to_plot), 6 * n_rows), nrows=n_rows, ncols=len(r_to_plot), layout='tight')

    for n in np.arange(n_rows):
        for i, (key, ax) in enumerate(zip(r_to_plot, axes[n])):
            file_name = dict_files[key][n]
            print(file_name)
            # file_name_prop = file_name.replace('output', 'prop').replace('.tree', '')
            # prop = utils.pickle_load(file_name_prop)

            class_mapping_df = pd.read_csv(map_files[key][0], sep=":", header=None, skiprows=1, names=["Index", "Class"])
            class_mapping = dict(class_mapping_df.values)

            G = convert_tree_to_nx(file_name)
            node_classes = [class_mapping[int(str(node).replace('N', ''))] for node in G.nodes()]

            # Plot del grafo con colori basati sulle classi
            # pos = nx.spring_layout(G, seed=1)  # posizionamento dei nodi
            # nx.draw(G, pos, with_labels=False, node_color=node_classes, cmap=plt.cm.tab10, node_size=300, ax=ax)
            if is_chain_1d(G):
                root_node = list(G.nodes(data=True))[0][0]
            else:
                root_node = list(G.nodes(data=True))[1][0]

            plot_hierarchical_tree(G,
                                   root=root_node,
                                   ax=ax,
                                   # fig_size=(12, 8),
                                   node_size=100,
                                   with_labels=False,
                                   node_color=node_classes,
                                   arrows=False,
                                   cmap=plt.cm.tab10)
            # plt.show()
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.tab10,
    #                            norm=plt.Normalize(vmin=min(node_classes), vmax=max(node_classes)))
    # sm._A = []
    # cbar = plt.colorbar(sm, ax=ax, ticks=range(min(node_classes), max(node_classes) + 1), label='Class')
    # cbar.ax.tick_params(labelsize=20)  # Imposta la dimensione del carattere delle tick labels
    # cbar.ax.set_ylabel('Class', fontsize=24, fontweight='bold')

    plt.savefig(save_dir+'hierarchical_network_trasformation_r{:s}_{:d}.{:s}'.format(name_fig, G.number_of_nodes(), fig_format), dpi=300)
    plt.show()
