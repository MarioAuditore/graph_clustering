import numpy as np
import networkx as nx
import igraph as ig

from heapq import heappop, heappush
from itertools import count

from .utils import filtration
from tqdm import tqdm

# from sklearn.cluster import HDBSCAN, KMeans
# from gudhi.clustering.tomato import Tomato
# import leidenalg as la

from copy import deepcopy
from itertools import combinations


# --- Topological clustering ---
# Triangulation: this func adds edges to closest nodes, found by dijkstra.
# It's purpose is to add more triangles in the graph
def dijkstra_neighbourhood(graph: nx.Graph,
                 start: int,
                 threshold: float,
                 max_degree: int = None,
                 weight: str = 'length'):
    
    if (threshold == 0) or (max_degree and graph.degree[start] >= max_degree):
        return 
    
    adjacency = graph._adj
    nodes = graph.nodes()
    c = count()
    push = heappush
    pop = heappop
    dist = {}
    pred = {}
    fringe = []
    push(fringe, (0.0, next(c), 0, start, None))
    while fringe:
        (d, _, n, v, p) = pop(fringe)
        if v in dist:
            continue
        dist[v] = (d, n)
        pred[v] = p
        
        for u, e in adjacency[v].items():
            vu_dist = d + e[weight]
            if vu_dist < threshold:
                if u not in dist:
                    push(fringe, (vu_dist, next(c), n + 1, u, v))
    
    dist = sorted(dist.items(), key=lambda arr : arr[-1])
    
    while dist:
        if max_degree and graph.degree[start] >= max_degree:
            break
        u = dist.pop(0)[0]
        if u != start and [start, u] not in graph.edges:
            graph.add_edge(start, u, length=vu_dist)


# Function for building graph of k-nearest neighbours
def dijkstra_knn(graph: nx.Graph,
                 graph_knn: nx.Graph,
                        start: int,
                        k: int,
                        weight:  str = 'length'
                 ):
    adjacency = graph._adj
    c = count()
    push = heappush
    pop = heappop
    dist = {}
    fringe = []
    
    dist[start] = 0.0
    push(fringe, (0.0, next(c), start))
    visited = set()
    counter = 0
    while fringe:
        (d, _, v) = pop(fringe)
        # Add closest node except for the start point
        if counter > 0:
            if counter <= k:
                graph_knn.add_edge(start, v, length=d)
                counter += 1
            else:
                return
        else:
            counter += 1
        # Find closest nodes in the neighbourhood
        for u, e in adjacency[v].items():
            vu_dist = d + e[weight]
            if u not in dist or dist[u] > vu_dist:
                dist[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))

# Function wrap
def build_knn_graph(G, k, weight='length'):
    G_knn = nx.Graph()
    for v in G.nodes:
        dijkstra_knn(G, G_knn, v, k, weight=weight)
    return G_knn
        

# Find the shortest path to a set of nodes 
# and return the id of the closest node from set
def dijkstra_pfa_to_set(graph: nx.Graph,
                        start: int,
                        ends: set[int],
                        weight: str
                 ) -> \
        tuple[float, list[int]]:
    adjacency = graph._adj
    c = count()
    push = heappush
    pop = heappop
    dist = {}
    fringe = []
    
    dist[start] = 0.0
    push(fringe, (0.0, next(c), start))
    visited = set()
    while fringe:
        (d, _, v) = pop(fringe)
        if v in ends:
            return v
        if len(visited) == len(ends):
            break
        for u, e in adjacency[v].items():
            vu_dist = d + e[weight]
            if u not in dist or dist[u] > vu_dist:
                dist[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
        

# Function to merge triangles
def simplex_merging(simplex_set, edges):
    '''
    Function that merges abstract sets, based on common edge

    Parameters
    ----------
    simplex_set : list[set[int]]
                  Set of simplexes to merge by common edges.
    edges : list[set[int]]
            Set of existing edges for check.
    '''
    # TODO Возможно ввести параллелизм: случайно склеиваем треугольники
    simplex_set = deepcopy(simplex_set)

    clusters = []
    visited = []

    # Function to find matches
    def find_match(t_1):
        match_found = False
        # Find matching triangles
        for t_2 in simplex_set:
            if t_2 in visited:
                continue
            # If we have more than 2 common vertices
            if len(t_1.intersection(t_2)) > 1:
                # In case of triangles that means a common edge
                if (len(t_1) == 3) or (len(t_2) == 3):
                    t_1.update(t_2)
                    visited.append(t_2)
                    match_found = True
                else:
                    # In case of larger sets, we need to check that there really is a common edge
                    for edge in combinations(list(t_1.intersection(t_2)), r=2):
                        if set(edge) in edges:
                            t_1.update(t_2)
                            visited.append(t_2)
                            match_found = True
                            break
        return match_found

    for t_1 in simplex_set:
        if t_1 in visited:
            continue
        else:
            visited.append(t_1)
        match_found = find_match(t_1)
        # Re-scan the set and find new matching triangles
        while match_found:
            match_found = find_match(t_1)
        
        clusters.append(t_1)
    
    return clusters


# Clustering algorithm
def filtration_merging(G, q=[0.1 * i for i in range(1, 11)], weight='length'):
    '''
    Clusterization based on filtration process and simplex merging.\
    Provided with graph G and array of quantiles q, this function constructs\
    a subgraph, whose edges are less than current threshold. Thresholds are obtained by taking\
    quantiles of edge lengths corresponding to q.

    Parameters
    ----------
    G : networkx.Graph
        Graph to clusterize
    weight : str
             alias for obtaining weights of edges
    q : list[int]
        Array of quantiles
    '''
    # For the core of clusters
    clusters_base = []
    # For all clusterizations
    cluster_base_set = []

    # Get birth times of simplexes
    simplicial_complex = filtration(G, weight=weight)

    # TODO: в качестве базы кластеров можно использовать тетраэдры и тд
    dim_base = 2
    simplex_set = {}
    for subcomplex in simplicial_complex[dim_base:]:
        simplex_set = simplex_set | subcomplex

    # triangles = sorted(triangles, key=lambda arr : arr[-1])

    # Get all birth times of triangles
    levels = np.array([simplex_set[s] for s in simplex_set])
    # Choose the subset according to provided quantiles
    thresholds = np.quantile(levels, q=q)

    prev_level = 0

    for i in tqdm(range(len(thresholds))):
        # Find edges at current threshold
        edges = [set(e) for e in simplicial_complex[1] if simplicial_complex[1][e] <= thresholds[i]]

        if i == 0:
            new_simplexes = [set(s) for s in simplex_set if simplex_set[s] <= thresholds[i]]
        else:
            new_simplexes = [set(s) for s in simplex_set if (thresholds[i - 1] < simplex_set[s]) and (simplex_set[s] <= thresholds[i])]

        # TODO: perhaps k_clique_percolation is faster
        # Perform clustering: merge(clusters + merge(new_triangles))
        clusters_base = simplex_merging(clusters_base + simplex_merging(deepcopy(new_simplexes), edges), edges)
        
        # Save the core of clusterization produced by triangles
        cluster_base_set.append(clusters_base)
    
    return cluster_base_set