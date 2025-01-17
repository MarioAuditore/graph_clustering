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

