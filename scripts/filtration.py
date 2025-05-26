from copy import copy
from random import random, seed
from itertools import combinations
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import igraph as ig
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
        # TODO: Replace internal graph with igraph?
        self.graph = graph.copy()
        self.seed = seed
        self.weight = weight
        self.distance_distribution = [d[weight]
                                      for _, _, d in graph.edges(data=True)]

        # Set clique information
        self.k = clique_size
        self.triangulation = self._compute_triangulation()
        self.triangle_cobound = self._compute_cobound()

        # Map original nodes to integer IDs and back
        self.node_to_id = {node: i for i, node in enumerate(graph.nodes())}
        self.id_to_node = {i: node for node, i in self.node_to_id.items()}

        # Compute shortest path distance matrix
        self.dist_matrix = self._compute_shortest_path_distances()

        # Precompute layout for visualization (using original graph)
        self.pos = nx.spring_layout(self.graph, seed=self.seed)

        # Storage for clustering results: dict of eps -> dict node->community
        self.clusterings = {}

    def _compute_triangulation(self,):
        # Find all triangles
        id_to_node = {id: node for id, node in enumerate(self.graph.nodes)}
        g = ig.Graph.from_networkx(self.graph)
        triangulation = [tuple([id_to_node[v] for v in t])
                         for t in g.list_triangles()]

        clique_birth = {}
        for t in triangulation:
            max_weight = 0
            for u, v in combinations(t, 2):
                if self.graph.has_edge(u, v):
                    max_weight = max(
                        max_weight, self.graph[u][v][self.weight])
            clique_birth[t] = max_weight
        # Save results
        return clique_birth

    def _compute_cobound(self,):
        co_bound = {}
        for t in self.triangulation:
            for e in combinations(t, 2):
                e = tuple(sorted(e))
                if e in co_bound:
                    co_bound[e].add(tuple(sorted(t)))
                else:
                    co_bound[e] = {tuple(sorted(t))}
        return co_bound

    def _compute_shortest_path_distances(self):
        lengths = dict(nx.all_pairs_dijkstra_path_length(
            self.graph, weight=self.weight))

        dist_matrix = {}
        for v in self.graph.nodes:
            dist_matrix[v] = {}
            for u in self.graph.nodes:
                if v == u:
                    dist_matrix[v][u] = 0
                elif u in lengths[v]:
                    dist_matrix[v][u] = lengths[v][u]
                else:
                    # If no path found
                    dist_matrix[v][u] = np.inf

        return dist_matrix

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
        G = self.graph
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

        if metric == 'modularity':
            score = nx.community.modularity(
                self.graph,
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

        # Get list of feasible triangles
        triangle_queue = set(
            [tuple(sorted(t)) for t, d in self.triangulation.items() if d <= epsilon])

        communities = []

        while triangle_queue:
            # Start with some triangle
            start = triangle_queue.pop()
            # Cluster will be formed around it
            cluster = set(start)
            # Get adjacent triangles
            neighbours_queue = [t for e in combinations(
                start, 2) for t in self.triangle_cobound[e] if t in triangle_queue]
            # Remove them from the queue
            for t in neighbours_queue:
                triangle_queue.remove(t)
            # print(f'Start: {start} | next: {neighbours_queue}')
            # Iterate over neighbours
            while neighbours_queue:
                # Get new triangle
                next_triangle = neighbours_queue.pop()
                # Add it to cluster
                cluster.update(set(next_triangle))
                # Append new unseen triangles
                for e in combinations(next_triangle, 2):
                    # Check if we have not visited it already
                    for t in self.triangle_cobound[e]:
                        if t in triangle_queue:
                            # print(f'Cluster: {cluster} | next: {neighbours_queue}')
                            neighbours_queue.append(t)
                            # Remove them from the queue
                            triangle_queue.remove(t)
            # Add merged cluster to communities
            communities.append(cluster)

        # Map node -> community index
        node_to_comm = self._get_clustering(communities)

        # Assign unclustered nodes using original graph distances
        unclustered_nodes = set(self.graph.nodes) - set(node_to_comm.keys())

        # return G_eps, communities, unclustered_nodes, node_to_comm

        for node in unclustered_nodes:
            min_dist = np.inf
            closest_comm = None
            for comm_idx, comm in enumerate(communities):
                # Use precomputed distance matrix for node-to-community distances
                dist_to_comm = min(
                    self.dist_matrix[node][member] for member in comm)
                if dist_to_comm < min_dist:
                    min_dist = dist_to_comm
                    closest_comm = comm_idx
            if closest_comm is not None and min_dist < np.inf:
                node_to_comm[node] = closest_comm
                communities[closest_comm].add(node)
            else:
                node_to_comm[node] = len(communities)
                # communities.append({node})

        # Store clustering
        if save_clustering:
            self.clusterings[epsilon] = node_to_comm

        # Release memory
        # del G_eps

        return node_to_comm

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

    def cluster(self, metric: str = 'modularity', quantiles: list = np.linspace(1e-1, 1, 100), n_jobs: int = 1, verbose: bool = True):
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
        for node in self.graph.nodes():
            node_comm = clustering[node]
            for neighbor in self.graph._adj[node]:
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
                    break  # Restart iteration after any improvement

            if not improved:
                break  # Stop if no improvement in this iteration

        return current_clustering

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
                       for node in self.graph.nodes()]

        plt.figure(figsize=(8, 6))

        if epsilon:
            plt.title(f'FiltrationClustering at ε={epsilon:.4f}, k={self.k}')
            edges = [(node_a, node_b) for node_a, node_b,
                     d in self.graph.edges(data=True) if d[self.weight] <= epsilon]
        else:
            plt.title('FiltrationClustering')
            edges = [(node_a, node_b) for node_a, node_b,
                     d in self.graph.edges(data=True)]

        nx.draw_networkx_nodes(self.graph, self.pos,
                               node_color=node_colors, node_size=300, cmap=color_map)
        # plot only filtration edges
        nx.draw_networkx_edges(self.graph, self.pos,
                               edgelist=edges, alpha=0.5)
        nx.draw_networkx_labels(self.graph, self.pos)

        plt.axis('off')
        plt.show()
