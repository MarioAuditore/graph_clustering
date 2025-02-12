import networkx as nx
import numpy as np
import igraph as ig


# Weird paper: https://arxiv.org/pdf/2005.04806
# Main reference: https://arxiv.org/pdf/1205.6233

# --- Utility functions ---

def total_metric_wrapper(G, clusters, metric, weight='length', aggreg=np.mean):
    scores = []
    for cluster in clusters:
        scores.append(metric(G, cluster, weight))
    return aggreg(scores)


def cluster_inner_weights(G, cluster, weight='length'):
    G_cluster = G.subgraph(cluster)
    if weight:
        return nx.adjacency_matrix(G_cluster, weight=weight).sum()
    else:
        return len(G_cluster.edges)


def cluster_border_weights(G, cluster, weight='length'):
    quantity = 0
    total_sum = 0.0
    adjacency = G._adj
    for v in cluster:
        for n in adjacency[v]:
            if n not in cluster:
                if weight:
                    total_sum += adjacency[v][n][weight]
                quantity += 1
    if weight:
        return total_sum
    else:
        return quantity


# (A) Scoring functions based on internal connectivity:

def internal_density(G, cluster, weight='length'):
    n_s = len(cluster)
    m_s = cluster_inner_weights(G, cluster, weight)
    return 2 * m_s / (n_s * (n_s - 1) + 1e-15)


def average_degree(G, cluster, weight='length'):
    n_s = len(cluster)
    m_s = cluster_inner_weights(G, cluster, weight)
    return 2 * m_s / (n_s + 1e-15)


def fraction_over_median_degree(G, cluster, weight='length'):
    d_m = np.median([d for _, d in G.degree()])
    n_s = len(cluster)
    # Find median degree
    G_cluster = G.subgraph(cluster)
    n_filtered = len([v for v in cluster if G_cluster.degree(v) > d_m])
    return n_filtered / n_s


def triangle_participation_ratio(G, cluster, weight='length'):
    n_s = len(cluster)
    G_cluster = G.subgraph(cluster)
    g = ig.Graph.from_networkx(G_cluster)
    n_t = len(g.list_triangles())
    return n_t / n_s


# (B) Scoring functions based on external connectivity:

def expansion(G, cluster, weight='length'):
    n_s = len(cluster)
    c_s = cluster_border_weights(G, cluster, weight)
    return c_s / n_s


def cut_ratio(G, cluster, weight='length'):
    n_s = len(cluster)
    c_s = cluster_border_weights(G, cluster, weight)
    return c_s / (n_s * (len(G.nodes) - n_s) + 1e-15)


# (C) Scoring functions that combine internal and exter- nal connectivity:

def relative_cluster_density(G, cluster, weight='length'):
    m_s = cluster_inner_weights(G, cluster, weight)
    c_s = cluster_border_weights(G, cluster, weight)
    return 1 + m_s / (c_s + 1e-10)


def conductance(G, cluster, weight='length'):
    m_s = cluster_inner_weights(G, cluster, weight)
    c_s = cluster_border_weights(G, cluster, weight)
    return c_s / (2 * m_s + c_s + 1e-15)


def normalized_cut(G, cluster, weight='length'):
    m_s = cluster_inner_weights(G, cluster, weight)
    c_s = cluster_border_weights(G, cluster, weight)
    if weight:
        m = sum([d[-1][weight] for d in G.edges(data=True)])
    else:
        m = len(G.edges)
    conductance = c_s / (2 * m_s + c_s + 1e-15)
    return c_s / (2 * (m - m_s) + c_s + 1e-15) + conductance


def separability(G, cluster, weight='length'):
    m_s = cluster_inner_weights(G, cluster, weight)
    c_s = cluster_border_weights(G, cluster, weight)
    return m_s / (c_s + 1e-10)

# ODF-based functions

def out_degree_fraction(G, v, cluster, weight):
    odf = 0
    for n in G._adj[v]:
        if n not in cluster:
            if weight:
                odf += G._adj[v][n][weight]
            else:
                odf += 1
    return odf / G.degree(v)


def max_odf(G, cluster, weight='length'):
    max_odf = 0
    for u in cluster:
        odf = out_degree_fraction(G, u, cluster, weight)
        if max_odf < odf:
            max_odf = odf
    return max_odf


def average_odf(G, cluster, weight='length'):
    n_s = len(cluster)
    odf_sum = 0
    for u in cluster:
        odf_sum += out_degree_fraction(G, u, cluster, weight)
    return odf_sum / n_s


def flake_odf(G, cluster, weight='length'):
    n_s = len(cluster)
    total_sum = 0
    for v in cluster:
        odf_count = 0
        odf_weight = 0
        for n in G._adj[v]:
            if n not in cluster:
                odf_count += 1
                if weight:
                    odf_weight += G._adj[v][n][weight]
        if odf_count < (G.degree(v) / 2):
            if weight:
                total_sum += odf_weight
            else:
                total_sum += odf_count
    return total_sum / n_s