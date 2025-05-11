from copy import copy
from random import random, seed
from typing import Optional

# import gudhi
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


class FiltrationClustering:
    def __init__(self,
                 graph: nx.Graph,
                 clique_size: int = 3,
                 seed: int = 42,
                 weight: str = 'length'):
        """
        Parameters:
        -----------
        graph : networkx.Graph
            Input weighted graph. Edge weights represent distances.
        max_filtration : float
            Maximum filtration value (epsilon) for Vietoris-Rips complex.
        clique_size : int
            Size of cliques (k) for clique percolation method.
        seed : int
            Random seed for layout reproducibility.
        weight : str
            String attribute to retrieve weight from edges.
        """
        self.original_graph = graph.copy()
        # self.reference_graph = None
        self.weight = weight
        self.distance_distribution = [d[weight]
                                      for _, _, d in graph.edges(data=True)]

        self.k = clique_size
        self.seed = seed

        # Map original nodes to integer IDs and back
        self.node_to_id = {node: i for i, node in enumerate(graph.nodes())}
        self.id_to_node = {i: node for node, i in self.node_to_id.items()}
        self.n = len(self.node_to_id)

        # Build internal graph with integer nodes and weighted edges
        self.int_graph = nx.Graph()
        for u, v, data in graph.edges(data=True):
            w = data.get(weight, 1.0)  # default weight 1.0 if missing
            self.int_graph.add_edge(
                self.node_to_id[u], self.node_to_id[v], weight=w)

        # Compute shortest path distance matrix
        self.dist_matrix = self._compute_shortest_path_distances()

        # Build Rips complex once
        # self.simplex_tree = self._build_rips_complex()

        # # Get all simplices with filtration values
        # self.all_simplices = list(self.simplex_tree.get_filtration())

        # Precompute layout for visualization (using original graph)
        self.pos = nx.spring_layout(self.original_graph, seed=self.seed)

        # Storage for clustering results: dict of eps -> dict node->community
        self.clusterings = {}

    def _compute_shortest_path_distances(self):
        lengths = dict(nx.all_pairs_dijkstra_path_length(
            self.int_graph, weight='weight'))
        dist_matrix = np.full((self.n, self.n), np.inf)
        for i in range(self.n):
            dist_matrix[i, i] = 0.0
            for j, dist in lengths[i].items():
                dist_matrix[i, j] = dist
        return dist_matrix

    # def _build_rips_complex(self):
    #     rips = gudhi.RipsComplex(
    #         distance_matrix=self.dist_matrix, max_edge_length=self.max_filtration)
    #     return rips.create_simplex_tree(max_dimension=2)

    def _cpm_score(self, communities: list, gamma: float = 1.0):
        """
        Compute CPM score for given communities.

        Parameters:
        -----------
        communities : list of iterables
            List of communities (each community is iterable of nodes).
        gamma : float
            Resolution parameter.

        Returns:
        --------
        float
            CPM score
        """
        # if self.reference_graph:
        #     G = self.reference_graph
        # else:
        G = self.original_graph
        # Map node -> community index (assign first community if overlaps)
        node_to_comm = self._get_clustering(communities)

        # Sum edge weights inside communities
        e_c = [0.0] * len(communities)
        n_c = [len(comm) for comm in communities]

        for u, v, data in G.edges(data=True):
            w = data.get(self.weight, 1.0) if self.weight else 1.0
            if node_to_comm.get(u) == node_to_comm.get(v):
                c = node_to_comm[u]
                e_c[c] += w

        # Compute CPM score
        cpm_score = 0.0
        for i in range(len(communities)):
            possible_edges = n_c[i] * (n_c[i] - 1) / 2
            cpm_score += e_c[i] - gamma * possible_edges

        return cpm_score

    def _evaluate_clustering(self, clustering: dict, metric: str = 'modularity') -> float:
        '''
        Evaluate provided clustering
        '''
        # if self.reference_graph:
        #     G = self.reference_graph
        # else:
        G = self.original_graph

        if metric == 'modularity':
            score = nx.community.modularity(
                G,
                self._get_communities(clustering),
                weight=self.weight)
        elif metric == 'cpm':
            score = self._cpm_score(
                self._get_communities(clustering))
        else:
            raise Exception(
                f"Provided metric {metric}, valid ones are: 'modularity', 'cpm' ")
        return score

    def cluster_at(self, quantile: float, save_clustering: bool = True):
        """
        Perform clustering at filtration threshold, which is computed as a quantile of distance distribution.
        Returns a dict mapping original node labels to community indices.

        Parameters:
        -----------
        quantile : float
            Filtration threshold, for which clustering is computed. Value should lie in [0,1].
        save_clustering: bool
            Flag whether to save clustering or not. Turned off, when computing clusterings in parallel.

        Returns:
        --------
        list
            Clustering at the given epsilon

        """
        epsilon = np.quantile(self.distance_distribution, quantile)
        # No need to re-compute
        if epsilon in self.clusterings:
            return self.clusterings[epsilon]

        # Filter simplices up to epsilon and dimension k (cliques up to size k)
        # simplices_at_eps = [
        #     s for s in self.all_simplices if s[1] <= epsilon and len(s[0]) <= self.k]

        # Extract edges for graph construction
        # edges = [tuple(s[0]) for s in simplices_at_eps if len(s[0]) == 2]
        edges = [(self.node_to_id[node_a], self.node_to_id[node_b]) for node_a, node_b,
                 d in self.original_graph.edges(data=True) if d[self.weight] <= epsilon]

        # Build graph at this filtration
        G_eps = nx.Graph()
        G_eps.add_nodes_from(range(self.n))
        G_eps.add_edges_from(edges)

        # # Build graph copy
        # G_eps = self.original_graph.copy()
        # # Remove edges with weight > epsilon
        # edges_to_remove = [(u, v) for u, v, d in G_eps.edges(data=True)
        #                    if d.get(self.weight, 1.0) > epsilon]
        # G_eps.remove_edges_from(edges_to_remove)

        # Extract k-cliques directly from simplices
        # cliques_k = [tuple(s[0])
        #              for s in simplices_at_eps if len(s[0]) == self.k]

        # CPM with precomputed cliques
        # communities = [set(c) for c in nx.algorithms.community.k_clique_communities(
        #     G_eps, self.k, cliques=cliques_k)]
        communities = [
            set(c) for c in nx.algorithms.community.k_clique_communities(G_eps, self.k)]

        # Map node -> community index
        node_to_comm = self._get_clustering(communities)

        # Assign unclustered nodes using original graph distances
        unclustered_nodes = set(range(self.n)) - set(node_to_comm.keys())

        for node in unclustered_nodes:
            min_dist = np.inf
            closest_comm = None
            for comm_idx, comm in enumerate(communities):
                # Use precomputed distance matrix for node-to-community distances
                dist_to_comm = min(
                    self.dist_matrix[node, member] for member in comm)
                if dist_to_comm < min_dist:
                    min_dist = dist_to_comm
                    closest_comm = comm_idx
            if closest_comm is not None and min_dist < np.inf:
                node_to_comm[node] = closest_comm
                communities[closest_comm].add(node)
            else:
                node_to_comm[node] = len(communities)
                # communities.append({node})

        # Convert back to original node labels
        clustering = {self.id_to_node[node]
            : comm for node, comm in node_to_comm.items()}

        # Store clustering
        if save_clustering:
            self.clusterings[epsilon] = clustering

        # Release memory
        del G_eps

        return clustering

    def _get_clustering(self, communities: list) -> dict:
        '''
        Convert list of sets to dict, mapping node to community label
        Parameters:
        -----------
        communities : list
            List of sets of nodes, representing communities
        '''

        seed(self.seed)

        # Map node -> community index
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                # Case of adjacent nodes
                if node not in node_to_comm:
                    node_to_comm[node] = i
                elif random() > 0.5:
                    node_to_comm[node] = i
        return node_to_comm

    def _get_communities(self, clustering: dict) -> list:
        """
        Convert dict of node_to_communities into a set of communities

        Parameters:
        -----------
        clustering : dict
            Map from node to community
        """
        # clustering = self.clusterings[epsilon]
        communities = {}
        for node, label in clustering.items():
            if label in communities:
                communities[label].append(node)
            else:
                communities[label] = [node]
        return [c for _, c in communities.items()]

    def optimal_epsilon(self, metric: str = 'modularity', verbose: bool = True):
        """
        Pick the best computed clustering with respect to modularity score.

        Parameters:
        -----------
        metric : str
            Metric to optimize: 'cpm' or 'modularity'

        Returns:
        --------
        list
            Best clustering, according to modularity
        """
        best_score = -np.inf
        best_epsilon = None
        if verbose:
            print('Finding best clustering')
            loop_range = tqdm(self.clusterings)
        else:
            loop_range = self.clusterings

        for eps in loop_range:
            score = self._evaluate_clustering(
                self.clusterings[eps], metric=metric)

            if score > best_score:
                best_score = score
                best_epsilon = eps
        if verbose:
            print(
                f'Best threshold {best_epsilon:.4f} with score: {best_score}')

        return best_epsilon

    def cluster(self, metric: str = 'modularity', quantiles: list = np.linspace(1e-1, 1, 100), n_jobs: int = 7, verbose: bool = True):
        """
        Perform filtration clustering

        Parameters:
        -----------
        metric : str
            Metric to optimize: 'cpm' or 'modularity'
        quantiles : list[float]
            Quantiles, which will be used as thresholds for computing clusterings
        n_jobs : int
            Number of processes to use, while computing clustering in parallel

        Returns:
        --------
        list
            Best clustering, according to modularity
        """
        print('Performing Filtration Clustering')
        epsilons = np.quantile(self.distance_distribution, quantiles)
        # Compute in parallel all clusterings
        if verbose:
            loop_range = tqdm(quantiles)
        else:
            loop_range = quantiles
        clusterings = Parallel(n_jobs=n_jobs)(delayed(
            lambda q: self.cluster_at(q, save_clustering=False)
        )(q) for q in loop_range)

        # Save the results internally
        for i, epsilon in enumerate(epsilons):
            self.clusterings[epsilon] = clusterings[i]
        # Return best result
        eps = self.optimal_epsilon(metric=metric, verbose=verbose)
        optimal_clustering = self.clusterings[eps]
        # Enhance with heuristics
        optimal_clustering = self.border_heuristic(
            optimal_clustering, metric=metric)
        return self._get_communities(optimal_clustering)

    # --- Post-production ---
    def _find_border_nodes(self, clustering: dict):
        """
        Find nodes that have neighbors in other clusters.

        Parameters:
        -----------
        graph : nx.Graph
        clustering : dict {node: cluster_id}

        Returns:
        --------
            Dict of border nodes with their border clusters
        """
        border_nodes = {}
        for node in self.original_graph.nodes():
            node_comm = clustering[node]
            for neighbor in self.original_graph._adj[node]:
                if clustering[neighbor] != node_comm:
                    if node in border_nodes:
                        border_nodes[node].add(clustering[neighbor])
                    else:
                        border_nodes[node] = set([clustering[neighbor]])
        return border_nodes

    def border_heuristic(self, clustering, metric='modularity', max_iter=10):
        """
        Refine clustering by reassigning border nodes to neighboring clusters if quality improves.

        Parameters:
        -----------
        clustering : dict {node: cluster_id}
            Clustering to optimize
        metric : str
            Metric to optimize: 'cpm' or 'modularity'
        max_iter : int
            Maximum number of iterations

        Returns:
        --------
        dict {node: cluster_id} refined clustering
        """
        current_clustering = clustering.copy()

        current_quality = self._evaluate_clustering(
            current_clustering, metric=metric)

        for _ in range(max_iter):
            improved = False
            border_nodes = self._find_border_nodes(current_clustering)

            # Sort communities by size to prioritize small ones
            # small_comms = [comm for comm, nodes in sorted_comms if len(nodes) < 10]  # threshold example

            for node, border_communities in border_nodes.items():

                for new_comm in border_communities:
                    # Try moving node to new_comm
                    temp_clustering = current_clustering.copy()
                    temp_clustering[node] = new_comm

                    temp_quality = self._evaluate_clustering(
                        temp_clustering, metric=metric)

                    if temp_quality > current_quality:
                        current_clustering = temp_clustering
                        current_quality = temp_quality
                        improved = True
                        break  # Move on to next node after improvement

                if improved:
                    # print(f'Score improved to {current_quality}')
                    break  # Restart iteration after any improvement

            if not improved:
                break  # Stop if no improvement in this iteration

        return current_clustering

    # def clustering_local_filtration(self, clustering: dict, metric: str = 'modularity') -> dict:

    #     # Sort communities by their size
    #     communities = self._get_communities(clustering)
    #     communities = sorted(communities, key=len)[::-1]
    #     # Save original graph
    #     original_graph = self.original_graph
    #     for c in tqdm(communities):
    #         # Get subgraph, representing cluster
    #         G_cluster = self.original_graph.subgraph(c)
    #         self.original_graph = G_cluster
    #         subcluster_filtration = []
    #         # Filter cluster
    #         for q in np.linspace(0.1, 1, 100):
    #             temp_clustering = copy(clustering)
    #             subclustering = self.cluster_at(q, save_clustering=False)
    #             # Save result in format of whole clustering
    #             for node, label in subclustering.items():
    #                 temp_clustering[node] = label
    #             subcluster_filtration.append(temp_clustering)

    #         # Now evaluate on original graph
    #         self.original_graph = original_graph
    #         # Init optimal solution (no changes by default)
    #         optimal_clustering = copy(clustering)
    #         best_score = self._evaluate_clustering(clustering, metric)
    #         for temp_clustering in subcluster_filtration:
    #             # Check if performance got better
    #             temp_score = self._evaluate_clustering(temp_clustering, metric)
    #             if temp_score > best_score:
    #                 print(f'Improved to {temp_score}')
    #                 best_score = temp_score
    #                 optimal_clustering = temp_clustering
    #         # Update clustering
    #         clustering = optimal_clustering
    #     return clustering

    # --- Visualization ---

    def plot_clustering(self, epsilon: Optional = None, communities: Optional = None):
        """
        Visualize clustering at filtration epsilon.

        Parameters:
        -----------
        epsilon : float
            Filtration threshold, for which clustering is computed.
        communities : list
            List of sets of nodes, representing communities
        """

        if communities is not None:
            clustering = self._get_clustering(communities)

        elif epsilon is not None:
            if epsilon not in self.clusterings:
                raise ValueError(
                    f"Clustering at epsilon={epsilon} not computed yet. Run cluster_at(epsilon) first.")

            clustering = self.clusterings[epsilon]
        else:
            raise ValueError(
                'One of the parameters should be passed: eigher epsilon or communities')

        # Assign colors to communities
        communities = set(clustering.values())
        num_comms = len(communities)
        color_map = cm.get_cmap('tab20', num_comms)

        # Map community to color index
        comm_to_color = {comm: i for i, comm in enumerate(sorted(communities))}

        # Node colors in original graph order
        node_colors = [color_map(comm_to_color[clustering[node]])
                       for node in self.original_graph.nodes()]

        # Extract edges at epsilon (same as in cluster_at)
        # simplices_at_eps = [
        #     s for s in self.all_simplices if s[1] <= epsilon and len(s[0]) == 2]
        # edges_at_eps = [tuple(s[0]) for s in simplices_at_eps]
        plt.figure(figsize=(8, 6))

        if epsilon:
            plt.title(f'FiltrationClustering at ε={epsilon:.4f}, k={self.k}')
            edges = [(node_a, node_b) for node_a, node_b,
                     d in self.original_graph.edges(data=True) if d[self.weight] <= epsilon]
        else:
            plt.title('FiltrationClustering')
            edges = [(node_a, node_b) for node_a, node_b,
                     d in self.original_graph.edges(data=True)]

        nx.draw_networkx_nodes(self.original_graph, self.pos,
                               node_color=node_colors, node_size=300, cmap=color_map)
        # plot only filtration edges
        nx.draw_networkx_edges(self.int_graph, self.pos,
                               edgelist=edges, alpha=0.5)
        nx.draw_networkx_labels(self.original_graph, self.pos)

        plt.axis('off')
        plt.show()
