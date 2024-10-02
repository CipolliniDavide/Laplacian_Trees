import networkx as nx


def load_edges_from_file(file_name):
    with open(file_name, 'r') as file:
        edges = [line.strip().split(' -> ') for line in file]
    return edges


def convert_tree_to_nx(file_name: str):
    edges = load_edges_from_file(file_name)
    # Create a directed graph
    G = nx.DiGraph()
    # Add edges to the graph
    G.add_edges_from(edges)
    # Check if the graph is directed
    if nx.is_directed(G):
        # Convert to undirected graph
        G = G.to_undirected()
    return G
