import os

import networkx as nx
import numpy as np
from random import choice, sample


# оптимальное количество кластеров из статьи
def get_opt_cluster_count(nodes: int) -> int:
    alpha = 8.09 * (nodes ** (-0.48)) * (1 - 19.4 / (4.8 * np.log(nodes) + 8.8)) * nodes
    return int(alpha)


def validate_cms(H: nx.Graph, communities: list[set[int]] | tuple[set[int]]) -> list[set[int]]:
    '''
    Function to process clustered graph.\
    Adds cluster data to the original graph data structure
    '''
    cls = []
    for i, c in enumerate(communities):
        for n in nx.connected_components(H.subgraph(c)):
            cls.append(n)
    for i, ids in enumerate(cls):
        for j in ids:
            H.nodes()[j]['cluster'] = i
    # assert nx.community.is_partition(H,cls)
    return cls

def get_node_for_initial_graph_v2(graph: nx.Graph):
    nodes = list(graph.nodes())
    f, t = choice(nodes), choice(nodes)
    while f == t:
        f, t = choice(nodes), choice(nodes)
    return f, t


def get_path(folder: str, name: str):
    '''
    Utility function to get path to store loaded graphs of cities.
    '''
    if not os.path.exists('./data'):
        os.mkdir('./data')
    path = os.path.join('./data', folder)
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.join(path, name)


def cut_subgraph(G, subgraph_size = 300, weight = 'length'):

    G_subgraph = G.copy()

    while len(G_subgraph.nodes) > subgraph_size:
        # Pick a node to remove
        v = sample(list(G_subgraph.nodes), 1)[0]
        # If it actually has neighbourhood
        if G_subgraph.degree[v] != 1:
            # Remove all neighbours with degree 1
            hanging_nodes = [n for n in G_subgraph._adj[v] if G_subgraph.degree[n] == 1]
            for n in hanging_nodes:
                G_subgraph.remove_node(n)
        
            # If it breaks connectivity
            neighbourhood = G_subgraph._adj[v]
            # TODO Ugly code, but uses nx.has_path
            G_buf = G_subgraph.copy()
            G_buf.remove_node(v)
            for n_1 in neighbourhood:
                for n_2 in neighbourhood:
                    if n_1 != n_2:
                        # If only one path exists
                        if not nx.has_path(G_buf, n_1, n_2):
                            # Compute the distance between them
                            edge_weigh = nx.path_weight(G_subgraph, (n_1, v, n_2), weight=weight)
                            # Add an edge
                            G_buf.add_edge(n_1, n_2)
                            G_subgraph.add_edge(n_1, n_2)
                            G_subgraph[n_1][n_2].update({weight : edge_weigh})
            del G_buf
        # Finally remove the node from Graph
        G_subgraph.remove_node(v)

    return G_subgraph