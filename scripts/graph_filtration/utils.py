import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from tqdm import tqdm
import igraph as ig


def plot_simplex(pos, K, show=False, alpha=0.7, ax=None):
    '''
    Function to plot simplicial complex

    Parameters
    ----------
    pos : ...
          Coordinates for plotting graph
    K : list
        List containing siplicies of different dimensions
    '''
    # max dim of simplices
    max_dim = len(K)

    def stack(idx):
        ret = np.empty((0, 2))
        for _id in idx:
            point = pos[_id]
            # ret = np.vstack((ret, X[_id,:]))
            ret = np.vstack((ret, point))
        return ret

    # plot vertices
    points = np.array([pos[v] for v in K[0]])
    if ax:
        ax.scatter(points[:, 0], points[:, 1], alpha=alpha)
    else:
        plt.scatter(points[:, 0], points[:, 1], alpha=alpha)

    # plot edges
    if max_dim >= 2:
        for e in K[1]:
            (start_id, end_id) = e
            start_point = pos[start_id]
            end_point = pos[end_id]
            if ax:
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'c-', alpha=min(0.2, alpha - 0.2))
            else:
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'c-', alpha=min(0.2, alpha - 0.2))

    # plot triangles
    if max_dim >= 3:
        for triangle in K[2]:
            t = plt.Polygon(stack(triangle), color="blue", alpha=min(0.1, alpha - 0.5))
            if ax:
                ax.gca().add_patch(t)
            else:
                plt.gca().add_patch(t)

    if show:
        plt.show()


def check_simplex_boundaries(simplex : list, simplicial_set : dict) -> bool:
    '''
    Checks whether simplex exists in the given simplicial set\
    or not by checking the existence of all it's boundaries.

    Parameters
    ----------
    simplex : list
              Set of verticies, representing the simplex you want to check
    simplicial_set : dict[tuple, int]
                     Simplicial complex, where you want to check the existence
    '''
    k = len(simplex)
    boundaries = combinations(simplex, r=k-1)
    for b in boundaries:
        b = tuple(sorted(b))
        if b not in simplicial_set:
            return False
    return True


# TODO: На больших графах медленная функция -> кластеризация медлит на Москве 
# Нужна индексация/хеширование
# Узнал как можно индексировать: https://arxiv.org/pdf/1908.02518, page 6.
def find_simplex_birth(simplex, simplicial_complex):
    '''
    Given a simplex, finds it's time of birth

    Parameters
    ----------
    simplex : set[int]
              A set of vertices, representing a simplex
    simplicial_complex : dict[tuple, int]
                         Here we expect a dictionary which maps simplex to it's time of birth.

    '''
    # Find boundaries of a simplex
    boundaries = combinations(simplex, r=len(simplex)-1)
    birth = 0.0
    # Collect birth time of boundaries
    for b in boundaries:
        b = tuple(sorted(b))
        if b not in simplicial_complex:
            raise Exception(f'Got simplex {simplex} with non-existent face {b}')
        else:
            # Choose the latest one
            if birth < simplicial_complex[b]:
                birth = simplicial_complex[b]
    return birth


def construct_k_simplex(K, dim_k, verbose=True):
    '''
    Creates k-dimensional simplexes in given simplicial complex K\
    if possible

    Parameters
    ----------
    K : list[dict[tuple, int]]
        Simplicial complex - list of dictionaries, where each dict corresponds to simplexes of specific dimensions:\
        K[0] - vertices, K[1] - edges, K[2] - triangles, etc. Dictionaries map simplexes to their birth time.
    dim_k : int
            Maximum dimensionality of simplexes to build. 
    '''
    # To avoid duplicates
    seen_simplex = []
    # Set of boundaries of k-simplices
    k_faces = K[dim_k - 1]
    if len(K) - 1 < dim_k:
        K.append([])
    # Iterate over boundaries
    face_range = tqdm(k_faces) if verbose else k_faces
    for face_a in face_range:
        for face_b in k_faces:
            if face_b != face_a:
                # Find adjacent faces
                if len(set(face_a).intersection(set(face_b))) == dim_k - 1:
                    # Create potential candidates
                    k_simplex = set(face_a).union(set(face_b))
                    # Check if already seen
                    if k_simplex not in seen_simplex:
                        seen_simplex.append(k_simplex)
                        # Check if boundaries of this simplex are present in the simplicial complex
                        if check_simplex_boundaries(k_simplex, k_faces):
                            # Save new simplex and it's birth time
                            simplex = tuple(sorted(k_simplex))
                            K[dim_k][simplex] = find_simplex_birth(simplex, k_faces) #max(k_faces[face_a], k_faces[face_b])
    return K


def filtration(G, dim_k=2, weight='length'):
    '''
    Filtration function on given weighted graph

    Parameters
    ----------
    G : networkx.Graph
        Graph to perform filtration on
    dim_k : int
            Highest dimension of simplexes to search for
    weight : str
             The edge attribute that holds the numerical value used for the edge weight.
    
    Returns
    -------
    simplicial_complex : list[list]
                         Simplicial complex with birth moments. First array represents nodes, second - eges,\
                            third - triangles, etc.
    '''
    # Construct basic simplexes
    
    # K = [
    #     list(G.nodes), # dim = 0 : vertices
    #     [set(e) for e in G.edges], # dim = 1 : vertices
    #     [set([g.vs[v]['_nx_name'] for v in cycle]) for cycle in g.list_triangles()] # dim = 2 : triangles
    #     ]

    K = []
    # Init vertices
    vertices = {}
    for v in G.nodes:
        vertices[v] = 0.0
    K.append(vertices)

    # Init edges
    edges = {}
    for e in G.edges(data=True):
        edges[tuple(sorted(e[:-1]))] = e[-1][weight]
    K.append(edges)

    # Init triangles
    g = ig.Graph.from_networkx(G)
    triangles = {}
    for t in g.list_triangles():
        t_id = tuple(sorted(g.vs[v]['_nx_name'] for v in t))
        triangles[t_id] = find_simplex_birth(t_id, edges)
    K.append(triangles)

    # Find all simplexes
    if dim_k > 2:
        for k in range(3, dim_k + 1):
            K = construct_k_simplex(K, dim_k=k)
    
    return K

