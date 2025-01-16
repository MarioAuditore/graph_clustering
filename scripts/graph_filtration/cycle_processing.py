import numpy as np
import networkx as nx
import igraph as ig



# --- Topological Clustering ---

def igraph_cycle_processing(cycle, edges):
    '''
    Utils function, which represents igraph cycles as sets of nodes
    '''
    node_set = []
    for edge in cycle:
        (node_a, node_b) = edges[edge]
        if node_a not in node_set:
            node_set.append(node_a)
        if node_b not in node_set:
            node_set.append(node_b)
    return node_set


def triangulate(cycle):
    '''
    Triangulates cycle of any length. Returns array of edges\
    to add for triangulation.
    
    Parameters
    ----------
    cycle : list
            Array of vertices, representing your cycle
    '''

    trinagulation_base = cycle[::2]
    add_edges = []
    # get basic triangulation
    for i, v in enumerate(trinagulation_base):
        if i == len(trinagulation_base) - 1:
            add_edges.append(set([v, trinagulation_base[0]]))
        else:
            add_edges.append(set([v, trinagulation_base[i+1]]))
    # Now triangulate the figure inside if needed
    if len(trinagulation_base) > 3:
        add_edges += triangulate(trinagulation_base)
    # check if there are no duplicates
    unique_edges = []
    for edge in add_edges:
        if edge not in unique_edges:
            unique_edges.append(edge)
    return unique_edges

def triangulate_graph(cycle_basis):
    add_edges = []
    for cycle in cycle_basis:
        add_edges += triangulate(cycle)
    return add_edges

def get_cycle_basis(G):
    '''
    Gets minimum cycle basis from igraph
    '''
    g = ig.Graph.from_networkx(G)

    ig_min_cycles = g.minimum_cycle_basis()
    ig_edges = g.get_edgelist()

    ig_min_cycles = [igraph_cycle_processing(cycle, ig_edges) for cycle in ig_min_cycles]
    cycle_basis = [[g.vs[v]['_nx_name'] for v in cycle] for cycle in ig_min_cycles]
    return cycle_basis


def cycle_iterator(cycle):
    for i, left_v in enumerate(cycle):
        if left_v == cycle[-1]:
            right_v = cycle[0]
        else:
            right_v = cycle[i+1]
        yield (left_v, right_v)
        
def get_cycle_length(G, cycle, weight='weight'):
    length = 0.0
    for (start_id, end_id) in cycle_iterator(cycle):
        length += G[start_id][end_id][weight]
    return length