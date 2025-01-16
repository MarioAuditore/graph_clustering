import os

import networkx as nx
import numpy as np
import random


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
    f, t = random.choice(nodes), random.choice(nodes)
    while f == t:
        f, t = random.choice(nodes), random.choice(nodes)
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
