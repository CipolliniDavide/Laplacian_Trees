import networkx as nx
import numpy as np
# from community import community_louvain
from sklearn.preprocessing import PowerTransformer

from . import utils


def compute_degree_correlation_matrix(graphs, separate_matrices=True):
    """
    Compute the degree correlation matrix of a graph or a list of graphs.
    Each element represents the probability of finding
    two nodes of degree k and k' connected by a link.

    Parameters:
        graphs (networkx.Graph or list): The input graph or a list of graphs.
        separate_matrices (bool): If True, return a list of separate degree correlation matrices.

    Returns:
        degree_corr_matrix (numpy.ndarray or list): The degree correlation matrix or a list of matrices.
    """
    if isinstance(graphs, list):  # Handle a list of graphs
        max_degree = max(max(dict(G.degree()).values()) for G in graphs)
        if separate_matrices:
            degree_corr_matrices = []

            for graph in graphs:
                degree_corr_matrix = np.zeros((max_degree + 1, max_degree + 1))
                for u, v in graph.edges():
                    degree_u = graph.degree[u]
                    degree_v = graph.degree[v]
                    degree_corr_matrix[degree_u][degree_v] += 1
                    degree_corr_matrix[degree_v][degree_u] += 1

                degree_corr_matrix /= (2 * graph.number_of_edges())
                degree_corr_matrices.append(degree_corr_matrix)

            return degree_corr_matrices

        else:
            degree_corr_matrix = np.zeros((max_degree + 1, max_degree + 1))

            for graph in graphs:
                for u, v in graph.edges():
                    degree_u = graph.degree[u]
                    degree_v = graph.degree[v]
                    degree_corr_matrix[degree_u][degree_v] += 1
                    degree_corr_matrix[degree_v][degree_u] += 1

            degree_corr_matrix /= (2 * sum(G.number_of_edges() for G in graphs))

    else:  # Handle a single graph
        max_degree = max(dict(graphs.degree()).values())
        degree_corr_matrix = np.zeros((max_degree + 1, max_degree + 1))

        for u, v in graphs.edges():
            degree_u = graphs.degree[u]
            degree_v = graphs.degree[v]
            degree_corr_matrix[degree_u][degree_v] += 1
            degree_corr_matrix[degree_v][degree_u] += 1

        degree_corr_matrix /= (2 * graphs.number_of_edges())

    return degree_corr_matrix



def degree_correlation_function(graphs, separate_matrices=False):
    '''
    Computes the conditional probability P(k'|k) = P(k,k')/P(k)
    and the computes the degree correlation function: knn(k) = Sum_k' k' * P(k'|k)

    :param (networkx.Graph or list): The input graph or a list of graphs.
    :param separate_matrices (bool): If True, compute separate matrices for each graph in the list.
                                    If False and if graphs is a list of graphs, it uses all the graphs in the list
                                    to compute the degree correlation matrix

    Returns:
        degree_range (list): List of node degrees.
        degree_correlation (list): knn(k) is the average degree of the neighbors of all degree-k nodes
    '''
    if isinstance(graphs, list):  # Handle a list of graphs
        if separate_matrices:
            degree_corr_matrices = compute_degree_correlation_matrix(graphs, separate_matrices=True)
            degree_range_list = []
            degree_correlation_list = []

            for graph, deg_corr_matx in zip(graphs, degree_corr_matrices):
                max_degree = max(dict(graph.degree()).values())
                degree_range = np.arange(max_degree + 1)
                row_sums = np.sum(deg_corr_matx, axis=1, keepdims=True)
                conditional_prob = deg_corr_matx / row_sums
                degree_corr_f = np.dot(conditional_prob, degree_range)
                degree_range_list.append(degree_range)
                degree_correlation_list.append(degree_corr_f)

            return degree_range_list, degree_correlation_list

        else:
            max_degree = max(max(dict(G.degree()).values()) for G in graphs)
            degree_corr_matrix = compute_degree_correlation_matrix(graphs, separate_matrices=False)
            row_sums = np.sum(degree_corr_matrix, axis=1, keepdims=True)
            conditional_prob = degree_corr_matrix / row_sums
            degree_range = np.arange(max_degree + 1)
            degree_corr_f = np.dot(conditional_prob, degree_range)

    else:  # Handle a single graph
        degree_corr_matrix = compute_degree_correlation_matrix(graphs)
        row_sums = np.sum(degree_corr_matrix, axis=1, keepdims=True)
        conditional_prob = degree_corr_matrix / row_sums
        max_degree = max(dict(graphs.degree()).values())
        degree_range = np.arange(max_degree + 1)
        degree_corr_f = np.dot(conditional_prob, degree_range)

    return degree_range, degree_corr_f




def plot_degree_correlation_matrix(degree_corr_matrix):
    """
    Plot the degree correlation matrix (normalized).

    Parameters:
        degree_corr_matrix (numpy.ndarray): The degree correlation matrix.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    plt.imshow(degree_corr_matrix, origin='lower', cmap='viridis')
    # plt.colorbar(label='Number of Pairs')
    plt.colorbar(label='Probability of edge between node1 and node2\n'+r'$P(k, k\prime)$')
    plt.xlabel('Degree of Node 2')
    plt.ylabel('Degree of Node 1')
    plt.title('Degree Correlation Matrix')
    plt.show()


def degree_clustering_mean_std(degrees, clustering_coefficients):
    """
    Calculate the unique degree values, their corresponding mean clustering coefficients,
    and the standard deviations of the mean clustering coefficients.

    Parameters:
        degrees (numpy.ndarray): Array of degree values.
        clustering_coefficients (numpy.ndarray): Array of clustering coefficient values.

    Returns:
        unique_degrees (numpy.ndarray): Array of unique degree values.
        mean_clustering (numpy.ndarray): Array of mean clustering coefficients.
        std_clustering (numpy.ndarray): Array of standard deviations of mean clustering coefficients.
    """
    unique_degrees = np.unique(degrees)
    mean_clustering = np.zeros_like(unique_degrees, dtype=float)
    std_clustering = np.zeros_like(unique_degrees, dtype=float)

    for i, degree in enumerate(unique_degrees):
        matching_indices = np.where(degrees == degree)
        matching_clustering = clustering_coefficients[matching_indices]
        mean_clustering[i] = np.mean(matching_clustering)
        std_clustering[i] = np.std(matching_clustering)

    return unique_degrees, mean_clustering, std_clustering




def edge_weight_distribution(G,
                             save_name=None,
                             save_fold="./",
                             title="",
                             cbar_label=""
                             ):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    weights_sequence = sorted(
        [d["weight"] for n1, n2, d in G.edges(data=True)], reverse=True
    )
    dmax = max(weights_sequence)

    fig = plt.figure("Edge weight analysis", figsize=(12, 8))

    ax0 = fig.add_subplot(121)
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])

    pos = nx.kamada_kawai_layout(G)
    # pos = nx.spring_layout(Gcc, seed=10396953)
    colors = [d["weight"] for n1, n2, d in G.edges(data=True)]
    vmin = min(colors)
    vmax = max(colors)
    print(vmin, vmax)
    # nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    # nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)

    ax0.set_title(
        "Largest connected component\nG has {} edges".format(G.number_of_edges())
    )
    ax0.set_axis_off()
    cmap = plt.cm.coolwarm
    options = {"node_size": 20, "alpha": 0.4}
    nx.draw(
        Gcc,
        pos=pos,
        ax=ax0,
        edge_color=colors,
        cmap=plt.cm.coolwarm,
        with_labels=False,
        vmin=vmin,
        vmax=vmax,
        **options,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar_ticks = np.linspace(vmin, vmax, 3, endpoint=True)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_ylabel("Weight")

    ax1 = fig.add_subplot(122)
    pdf, _, bins = utils.empirical_pdf_and_cdf(weights_sequence)
    ax1.plot(bins[1:], pdf, "b-", marker="o")
    ax1.set_title("Weight histogram")
    ax1.set_ylabel("Prob")
    ax1.set_xlabel("Weight")

    plt.tight_layout()
    if save_name:
        plt.savefig(save_fold + save_name + "WeightAnalysis.png")
        plt.close()
    else:
        plt.show()


def set_new_weights(graph, new_weights):
    edge_list = [(n1, n2) for n1, n2, weight in list(graph.edges(data=True))]
    # edge_weight_list = [weight['weight'] for n1, n2, weight in list(G.edges(data=True))]
    edge_list_with_attr = [
        (edge[0], edge[1], {"weight": w}) for (edge, w) in zip(edge_list, new_weights)
    ]
    graph.add_edges_from(edge_list_with_attr)


def pruning_of_edges(graph, p, save_fold="./"):
    def eps_edges(edge_weight_list, p=0.25):
        pdf, cdf, bins = utils.empirical_pdf_and_cdf(edge_weight_list, bins=100)
        try:
            return bins[cdf < p][-1]
        except:
            return 0

    weights = [w["weight"] for u, v, w in graph.edges(data=True)]
    # Find weights to keep: weights > epsilon
    epsilon = eps_edges(p=p, edge_weight_list=weights)
    # plot cdf
    pdf, cdf, bins = utils.empirical_pdf_and_cdf(weights, bins=100)
    # plt.close()
    # plt.plot(bins, cdf)
    # plt.axvline(epsilon, c='red')
    # plt.xlabel('Edge weight')
    # plt.ylabel('CDF')
    # plt.savefig(save_fold_figures + 'CDF_edges.png')
    # plt.close()
    #
    return [
        (u, v, {"weight": w["weight"]})
        for u, v, w in graph.edges(data=True)
        if w["weight"] > epsilon
    ]


def rescale_weights(graph, scale=(0.0001, 1), method="MinMax"):
    edge_list = [(n1, n2) for n1, n2, weight in list(graph.edges(data=True))]
    edge_weight_list = [
        weight["weight"] for n1, n2, weight in list(graph.edges(data=True))
    ]
    if method == "MinMax":
        edge_weight_list = utils.scale(edge_weight_list, scale)
    elif method == "box-cox":
        # print('Rescale method ', method)
        power = PowerTransformer(method="box-cox", standardize=True)
        data_trans = power.fit_transform(
            np.reshape(utils.scale(edge_weight_list, scale), newshape=(-1, 1))
        )
        edge_weight_list = utils.scale(data_trans.reshape(-1), scale)

    edge_list_with_attr = [
        (edge[0], edge[1], {"weight": w})
        for (edge, w) in zip(edge_list, edge_weight_list)
    ]
    graph.add_edges_from(edge_list_with_attr)


def connected_components(graph):
    """Return sub-graphs from largest to smaller"""
    sub_graphs = [
        graph.subgraph(c) for c in nx.connected_components(graph) if len(c) > 1
    ]
    sorted_sub_graphs = sorted(sub_graphs, key=len)
    return sorted_sub_graphs[::-1]


def relative_connection_density(graph, nodes):
    subG = graph.subgraph(nodes).copy()
    density = nx.density(subG)
    return density


def average_weighted_degree(graph, key_="weight"):
    """Average weighted degree of a graph"""
    edges_dict = graph.edges
    total = 0
    for node_adjacency_dict in edges_dict.values():
        total += sum(
            [adjacency.get(key_, 0) for adjacency in node_adjacency_dict.values()]
        )
    return total


def average_degree(graph):
    """Mean number of edges for a node in the network"""
    degrees = graph.degree()
    mean_num_of_edges = sum(dict(degrees).values()) / graph.number_of_nodes()
    return mean_num_of_edges


def filter_nodes_by_attr(graph, key_, key_value):
    """Returns the list of node indexes filtered by some value for the attribute key_"""
    return [
        idx for idx, (x, y) in enumerate(graph.nodes(data=True)) if y[key_] == key_value
    ]


def convert_to_networkx(node_list, edge_list_with_attr):
    graph = nx.Graph()
    graph.add_nodes_from(node_list)
    graph.add_edges_from(edge_list_with_attr)
    return graph



