import numpy as np
import networkx as nx
import igraph as ig

from heapq import heappop, heappush
from itertools import count

from .graph_filtration import cycle_processing

from . import utils
from tqdm import tqdm

from sklearn.cluster import HDBSCAN, KMeans
from gudhi.clustering.tomato import Tomato
import leidenalg as la

from copy import deepcopy


# --- Topological clustering ---

def simplex_collapse(simplex_set):
    '''
    Function that merges abstract sets, based on common edge
    '''
    simplex_set = deepcopy(simplex_set)

    clusters = []
    visited = []

    match_found = False

    # - брать случайно треугольники и склеивать их параллельно

    for t_1 in simplex_set:
        if t_1 in visited:
            continue
        # Find matching triangles
        for t_2 in simplex_set:
            if t_2 in visited:
                continue
            # тут не проверяется является ли пересечение ребром, 
            # достаточно просто проверить есть ли пересечение среди G.edges
            if len(t_1.intersection(t_2)) > 1:
                t_1.update(t_2)
                visited.append(t_2)
                match_found = True
        # Re-scan the set and find new matching triangles
        while match_found:
            match_found = False
            for t_2 in simplex_set:
                if t_2 in visited:
                    continue
                if len(t_1.intersection(t_2)) > 1:
                    t_1.update(t_2)
                    visited.append(t_2)
                    match_found = True
        clusters.append(t_1)
    
    return clusters


def simplicial_clustering(G, weight='length', levels=[0.5, 0.75, 0.9]):
    '''
    Function that performs full graph clusterization.

    Parameters
    ----------
    G : networkx.Graph
        Graph to perform clusterization on
    weight : str
             Edge attribute that stores weights.
    levels : list[float]
             Array of quantiles, which control the triangulation of cycles.\
             Basef on these levels we get different clusterizations.
    '''

    def dijkstra_closest_node(graph: nx.Graph,
                 start: int,
                 end: set[int],
                 cms: set[int] | None = None,
                 weight: str = 'length') -> \
        tuple[float, list[int]]:
        if start == end:
            return 0, [start]
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
           
            if v in end:
                break
            for u, e in adjacency[v].items():
                if cms and nodes[u]['cluster'] not in cms:
                    continue
                vu_dist = d + e[weight]
                if u not in dist:
                    push(fringe, (vu_dist, next(c), n + 1, u, v))
        d, n = dist[v]
        n += 1
        path = [None] * n
        i = n - 1
        e = v
        while i >= 0:
            path[i] = e
            i -= 1
            e = pred[e]
        return d, path
    
    # Init simplicial complex for triangulation
    G_triangulated = G.copy()
    g = ig.Graph.from_networkx(G_triangulated)
    K = [
            list(G.nodes), # dim = 0 : vertices
            list(G.edges), # dim = 1 : edges
            [set([g.vs[v]['_nx_name'] for v in cycle]) for cycle in list(g.list_triangles())] # dim = 2:
        ]
    
    # Get all basis cycles
    cycle_basis = cycle_processing.get_cycle_basis(G)
    # Sort them
    cycle_lengths = np.array([cycle_processing.get_cycle_length(G, cycle, weight=weight) for cycle in cycle_basis])
    cycle_basis = [data[0] for data in sorted(zip(cycle_basis, cycle_lengths), key=lambda data:data[-1])]
    # For new unseen cycles
    cycle_subset = []
    # For the core of clusters
    clusters_base = []
    # For all clusterizations
    clusters_set = []


    # Get clusterization on different levels
    for level in tqdm(np.quantile(cycle_lengths, q=levels)):
        # Add unseen cycles
        cycle_subset = [cycle_basis[i] for i in np.where(np.array(cycle_lengths) <= level)[0] if cycle_basis[i] not in cycle_subset]
        # cycle_subset = [cycle_basis[i] for i in range(int(len(cycle_basis) * level))]
        # Triangulate them
        new_edges = cycle_processing.triangulate_graph(cycle_subset)
        for edge in new_edges:
            if edge not in K[1]:
                K[1].append(edge)
                G_triangulated.add_edge(*edge)
        # Detect new triangles
        g = ig.Graph.from_networkx(G_triangulated)
        for cycle in list(g.list_triangles()):
            cycle = set([g.vs[v]['_nx_name'] for v in cycle])
            if cycle not in K[2]:
                # Save new triangle
                K[2].append(cycle)
                clusters_base.append(cycle)

        # clusters_base += triangles_set
        clusters_base = simplex_collapse(clusters_base)
        # triangles_set = []
        # clusters = simplex_collapse(K[2])
        
        # Cluster the rest nodes
        clusters = deepcopy(clusters_base)
        used_vertices = set()
        for cluster in clusters:
            used_vertices.update(cluster)

        unused_vertices = set(G.nodes) - used_vertices

        cluster_set = set()
        for c in clusters:
            cluster_set.update(c)
            
        for v in unused_vertices:
            _, path = dijkstra_closest_node(G, v, cluster_set, weight=weight)
            for id, c in enumerate(clusters):
                if path[-1] in c:
                    break
            clusters[id].add(v)

        
        clusters_set.append(clusters)

    return clusters_set



# --- Ordinary clustering algorithms ---

def louvain(H: nx.Graph, **params) -> list[set[int]]:
    '''
    Clustering by louvain communities
    '''
    communities = nx.community.louvain_communities(H,
                                                   seed=1534,
                                                   weight='length',
                                                   resolution=params['r'])
    return utils.validate_cms(H, communities)


def leiden(H: nx.Graph, **kwargs) -> list[set[int]]:
    '''
    Clustering by leiden algorithm - a modification of louvain
    '''
    # Leiden works with igraph framework
    G = ig.Graph.from_networkx(H)
    # Get clustering
    partition = la.find_partition(G, **kwargs)
    # Collect corresponding nodes
    communities = []
    for community in partition:
        node_set = set()
        for v in community:
            node_set.add(G.vs[v]['_nx_name'])
        communities.append(node_set)

    return utils.validate_cms(H, communities)


def hdbscan(H: nx.Graph, n_jobs: int = 7):
    def f(a,b):
        u = int(a[2])
        v = int(b[2])
        if (u,v) in H.edges() or (v,u) in H.edges():
            return H.edges()[(u,v)]['length']
        return float('inf')
    
    scan = HDBSCAN(metric=f, min_samples=1, max_cluster_size=30,n_jobs = n_jobs)
    x = np.array([[d['x'], d['y'], u] for u, d in H.nodes(data=True)])
    y = scan.fit_predict(x)
    communities = {}
    for i, u in enumerate(H.nodes):
        cls = y[i]
        if cls not in communities:
            communities[cls] = set()
        communities[cls].add(u)
    communities = [communities[cls] for cls in communities]
    del scan
    return utils.validate_cms(H, communities)


def kernighan_lin_bisection(g):
    cms = nx.community.kernighan_lin_bisection(g, weight='length')
    cms = utils.validate_cms(g, cms)
    res = []
    for c in cms:
        if len(c) < 100:
            res.append(c)
        else:
            gg = g.subgraph(c)
            rr = kernighan_lin_bisection(gg)
            rr = utils.validate_cms(gg, rr)
            res.extend(rr)
    return utils.validate_cms(g, res)


def k_clique_communities(g, k =2):
    cms= nx.community.k_clique_communities(g,k )
    res = utils.validate_cms(g, cms)
    assert nx.community.is_partition(g, res)
    return res


def girvan_newman(g):
    comp = nx.community.girvan_newman(g)
    for i, communities in tqdm(enumerate(comp), total= 800):
        cms = communities
        if len(cms)>800:
            break
    return utils.validate_cms(g, cms)


def tomato_resolver(H, r = 0.1):
    xx = {(d['x'], d['y']):u for u, d in H.nodes(data=True)}
    
    def f(a,b):
        
        u = a[2]
        v = b[2]
        
        if (u,v) in H.edges() or (v,u) in H.edges():
            return H.edges()[(u,v)]['length']
        return float('inf')
    
    x = np.array([[d['x'], d['y'], u] for u, d in H.nodes(data=True)])
        
    ex1 = Tomato(
        # metric = f,
            input_type="points",
        # n_jobs = 10,
        # p = 1,
            graph_type="radius",
            density_type="KDE",
            # n_clusters=800,
            r=r,
        )
    
    ex1.fit(x)
    cms = {}
    ll = list(ex1.labels_)

    for i, u in enumerate(H.nodes()):
        l = ll[i]
        if l not in cms:
            cms[l] = set()
        cms[l].add(u)
    
    cms_1 = []
    
    for c in cms.values():
        cms_1.append(c)
        
    return utils.validate_cms(H, cms_1)


# def resolve_by_mapper(H: nx.Graph):
#     id2node = {i:u for i, u in enumerate(H.nodes())}
#     mapper = km.KeplerMapper(verbose=0)
#     x = np.array([[d['x'], d['y']] for u, d in H.nodes(data=True)])
#     proj_data = mapper.fit_transform(x)
#     g = mapper.map(proj_data, x, clusterer= HDBSCAN())
#     cms = []
#     scan = HDBSCAN(metric=f, min_samples=1, max_cluster_size=30,n_jobs = 10)
#     x = np.array([[d['x'], d['y'], u] for u, d in g.nodes(data=True)])
#     
#     all_nodes = set()
#     for i, (k,v) in enumerate(dict(g['nodes']).items()):
#         # print(k)
#         for n in v:
#             all_nodes.add(n)
#         cms.append(set([id2node[id_node] for id_node in v ]))
#     # mapper.visualize(g, path_html="make_circles_keplermapper_output.html",
#     #              title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")
#     print(len(all_nodes), len(x))
#     return utils.validate_cms(H, cms)

