import networkx as nx
from tqdm import tqdm


def get_cluster_adjacency(graph: nx.Graph) -> dict[int: set[int]]:
    '''
    Get connections of neighbouring clusters and make a new adjacency dict
    '''
    _cls2n = {}
    # возможно обход в ширину эффективней - не будет дублей
    for u, u_data in graph.nodes(data=True):
        for v in graph[u]:
            v_data = graph.nodes()[v]
            if v_data['cluster'] == u_data['cluster']:
                continue
            c1 = v_data['cluster']
            c2 = u_data['cluster']
            if not (c1 in _cls2n):
                _cls2n[c1] = set()
            if not (c2 in _cls2n):
                _cls2n[c2] = set()
            _cls2n[c1].add(c2)
            _cls2n[c2].add(c1)
    return _cls2n


# cluster then yts point that are connected with neighboring clusters
# so seems like we store a set of border points
def get_cls2hubs(graph: nx.Graph) -> dict[int: set[int]]:
    '''
    Store points, located on borders of clusters, the ones that contain\
    neighbours from other clusters.
    '''
    _cls2hubs = {}
    for u, u_data in graph.nodes(data=True):
        for v in graph[u]:
            v_data = graph.nodes()[v]
            c1 = u_data['cluster']
            c2 = v_data['cluster']
            if c1 == c2:
                continue
            if not (c1 in _cls2hubs):
                _cls2hubs[c1] = set()
            if not (c2 in _cls2hubs):
                _cls2hubs[c2] = set()
            _cls2hubs[c1].add(u)
            _cls2hubs[c2].add(v)
    return _cls2hubs


def build_center_graph(
        graph: nx.Graph,
        communities: list[set[int]],
        cls2n: dict[int: set[int]],
        log: bool = False
) -> tuple[nx.Graph, dict[int, int]]:
    '''
    Function maps clusters in graph to their barycenters as representatives.\
    Then create new graph of barycenters.
    '''
    
    x_graph = nx.Graph()
    cls2c = {}
    iter = tqdm(enumerate(communities), total=len(communities), desc='find centroids') if log else enumerate(communities)
    for cls, _ in iter:
        gc = extract_cluster_list_subgraph(graph, [cls], communities)
        min_node = nx.barycenter(gc, weight='length')[0]
        u_data = graph.nodes()[min_node]
        x_graph.add_node(graph.nodes()[min_node]['cluster'], **u_data)
        cls2c[graph.nodes()[min_node]['cluster']] = min_node

    if len(x_graph.nodes) == 1:
        return x_graph, cls2c

    iter = tqdm(x_graph.nodes(), desc='find edges') if log else x_graph.nodes()

    # кажется networkx умеет сам все пары перебрать: 
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra.html
    for u in iter:
        for v in cls2n[u]:
            l = nx.single_source_dijkstra(graph, source=cls2c[u], target=cls2c[v], weight='length')[0]
            x_graph.add_edge(u, v, length=l)
    
    return x_graph, cls2c


def extract_cluster_list_subgraph(graph: nx.Graph, cluster_number: list[int] | set[int], communities=None) -> nx.Graph:
    '''
    Extraction of subgraph, created by specified communities
    '''
    if communities:
        return graph.subgraph(_iter_cms(cluster_number, communities))
    else:
        nodes_to_keep = [node for node, data in graph.nodes(data=True) if data['cluster'] in cluster_number]
    return graph.subgraph(nodes_to_keep)

def _iter_cms(cluster_number: list[int] | set[int], communities: list[set[int]]| tuple[set[int]]):
    '''
    Iterator to get set of nodes for set of specified communities
    '''
    for cls in cluster_number:
        for u in communities[cls]:
            yield u