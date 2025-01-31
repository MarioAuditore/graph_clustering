import networkx as nx
import numpy as np
import igraph as ig
import os
from collections import defaultdict

from .graph_filtration.utils import plot_simplex

from ortools.sat.python import cp_model
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from random import randint, sample, seed, shuffle


class TestCVRP:
    def __init__(self, G, max_steps, n_vehicles, weight='length', num_workers = None, time_limit = None, seed=0xAB0BA, verbose = False):
        
        self.G = G
        self.weight = weight
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.seed = seed
        self.P = n_vehicles
        self.verbose = verbose
        
        if num_workers:
            self.num_workers = num_workers
        else:
            self.num_workers = os.cpu_count() - 1


    def generate_demand(self, low=1, high=5):
        if self.seed:
            np.random.seed(self.seed)
        demands = {node: np.random.randint(low, high, 1)[0] for node in self.G.nodes}
        nx.set_node_attributes(self.G, demands, 'demand')
    

    def generate_capacities(self, demands):
        if self.seed:
            np.random.seed(self.seed)
            seed(self.seed)
        # Init vehicle capacities
        max_demand = max(demands)
        capacities = np.ones(self.P, dtype=int) * max_demand
        # Randomly distribute demand among vehicles
        # shuffle(demands)
        # step = int(len(demands) // self.P)
        # for i in range(self.P):
        #     capacities[i] += sum(demands[i * step : min((i + 1) * step, len(demands))])

        while max_demand * len(demands) >= capacities.sum():
            capacities[np.random.randint(0, self.P)] += max_demand 
        
        return capacities


    def find_solution(self, d_matrix, capacities, demands, hub_id = 0):
        '''
        Solve CVRP problem

        Params:
        -------
        d_matrix : numpy.array
                   Distance matrix of graph
        capacities : list
                     Capacities of vehicles
        demands : list
                  Demand of each vertex
        hub_id : int
                 id of a vertex, which serves as hub (it will have zero demand and will serve as starting point)

        Returns:
        --------
        np.array : Paths of CVRP solution
        float : total length of the solution
        '''
        # Define constraint model
        model = cp_model.CpModel()

        # defining variables
        Routes = {}  # x_ijp  - матрица маршрутов
        Y = {} # произведения ij

        MAX_STEPS = self.max_steps
        N = len(d_matrix)
        P = len(capacities)

        #____________________
        # Variables creation
        for p in range(P):
            for r in range(MAX_STEPS):
                for i in range(N):
                    Routes[p, r, i] = model.new_bool_var(name=f'rout{p}_{r}_{i}')

        for p in range(P):
            for r in range(MAX_STEPS - 1):
                for i in range(N):
                    for j in range(N):
                        Y[p, r, i, j] = model.new_bool_var(name=f'y_{p}_{r}_{i}_{j}')
                        if i != j:
                            # model.add_multiplication_equality(Y[p, r, i, j], [Routes[p, r, i], Routes[p, r + 1, j]])
                            model.AddBoolOr([Routes[p, r, i].Not(), Routes[p, r + 1, j].Not(), Y[p, r, i, j]])
                            model.AddImplication(Y[p, r, i, j], Routes[p, r, i])
                            model.AddImplication(Y[p, r, i, j], Routes[p, r + 1, j])
                        else:
                            model.add(Y[p, r, i, j] == 0)
        
        #_____________________
        # Constraints on capacity
        for p in range(P):
            model.add(sum(Routes[p, r, i] * demands[i] for i in range(N) for r in range(1, MAX_STEPS - 1)) <= capacities[p])

        #_________________________________
        # Constraints from adjacency matrix
        for i in range(N):
            for j in range(N):
                for r in range(MAX_STEPS - 1):
                    for p in range(P):
                        if d_matrix[i, j] == 0 and not (i == hub_id and j == hub_id):
                            model.add(Routes[p, r + 1, j] <= 1 - Routes[p, r, i])

        # x_iip = 0 - do not go from city to itself
        for i in range(N):
            if i != hub_id:
                for r in range(MAX_STEPS - 1):
                    for p in range(P):
                        model.add(Routes[p, r + 1, i] <= 1 - Routes[p, r, i])

        for p in range(P):
            # start in hub
            model.add(Routes[p, 0, hub_id] == 1)

            # Only one place at a time
            for r in range(MAX_STEPS):
                model.add(sum(Routes[p, r, i] for i in range(N)) == 1)  

            # If we came to hub, then we stay there
            for r in range(1, MAX_STEPS - 1):
                model.add(Routes[p, r + 1, hub_id] >= Routes[p, r, hub_id])  
            # We return to hub eventually
            model.add(sum(Routes[p, r, hub_id] for r in range(1, MAX_STEPS)) >= 1)

        # Ensure that every node is entered at least once
        for i in range(N):
            if i != hub_id:
                model.add(sum(Routes[p, r, i] for p in range(P) for r in range(1, MAX_STEPS)) >= 1)
                for p in range(P):
                    model.add(sum(Routes[p, r, i] for r in range(1, MAX_STEPS)) <= 1)

        # Minimize distance
        objective_func = sum(Y[p, r, i, j] * d_matrix[i, j] for r in range(MAX_STEPS - 1) for i in range(N) for j in range(N) for p in range(P))
        model.minimize(objective_func)
        
        solver = cp_model.CpSolver()
        solver.parameters.num_workers = self.num_workers
        solver.parameters.log_search_progress = self.verbose
        solver.parameters.max_time_in_seconds = 60.0 * self.time_limit # in minutes
        
        status = solver.solve(model)
        status_codes = {
            cp_model.UNKNOWN : 'UNKNOWN',
            cp_model.MODEL_INVALID : 'MODEL_INVALID',
            cp_model.FEASIBLE : 'FEASIBLE',
            cp_model.INFEASIBLE : 'INFEASIBLE',
            cp_model.OPTIMAL : 'OPTIMAL'
        }
        
    
        if status == 3:
            print(f'Solution status: {status_codes[status]}')
            print(f'capacities: {capacities} | {sum(capacities)}')
            print(f'demands: {demands} | {sum(demands)}')
            print(f'hub: {hub_id}')
            return None
        
        return np.reshape([solver.value(v) for v in Routes.values()], (P, MAX_STEPS , N)), solver.value(objective_func)


    def get_dist_matrix(self, G=None, subset=None):
        '''
        Get pairwise distance matrix from provided Graph
        '''
        # TODO: можно переехать цеиком на igraph
        if G is None:
            G = self.G
        g = ig.Graph.from_networkx(G)
        
        # re-numerate
        original_to_order = {}
        order_to_original = {}
        for i, id in enumerate(G.nodes):
            original_to_order[id] = i
            order_to_original[i] = id
        
        if subset:
            subset_ids = [original_to_order[id] for id in subset]
            d_matrix = np.array(g.distances(subset_ids, subset_ids, weights=self.weight))
            
            # re-numerate
            original_to_order = {}
            order_to_original = {}
            for i, id in enumerate(subset):
                original_to_order[id] = i
                order_to_original[i] = id
        else:
            d_matrix = np.array(g.distances(weights=self.weight))

        return d_matrix, order_to_original, original_to_order


    def test_cluster(self, clusters=None, hub_ids=None, subset_size=None):
        '''
        Solve CVRP for each cluster separately in given clusterization. \
        Capacities are generated to fit the demands, so that the task is feasible

        Params:
        -------
        clusters : list[set[int]]
                   List of sets containing ids of vertices
        hub_ids : list[int]
                  List of ids of vertices, which will serve as hubs

        Returns:
        --------
        total_length : float
                       Sum of total length of a solution in each cluster
        paths : list[list[int]]
                Paths of CVRP solution
        '''
        
        # Set seed
        np.random.seed(self.seed)
        seed(self.seed)
        # number of vehicles
        P = self.P

        if clusters is None:
            clusters = [list(self.G.nodes)]
            hub_ids = [0]
        
        paths = []
        total_length = 0
        
        for i, cluster in enumerate(tqdm(clusters)):
            # Get graph of cluster
            G_cluster = self.G.subgraph(cluster)
            # Choose a subset if needed
            subset = None
            if subset_size and len(cluster) > subset_size:
                subset = sample(list(cluster), subset_size)
                if hub_ids[i] not in subset:
                    subset[0] = hub_ids[i]
            # Get it's pairwise distance
            d_matrix, order_to_original, original_to_order = self.get_dist_matrix(G_cluster, subset)
            # Choose hub
            hub_id = original_to_order[hub_ids[i]]
            # Generate capacities and demands
            N = len(cluster)
            demands = np.array([data[-1]['demand'] for data in G_cluster.nodes(data=True)])
            demands[hub_id] = 0
            capacities = self.generate_capacities(demands)
            
            # Compute CVRP
            solution, length = self.find_solution(d_matrix, capacities, demands, hub_id)
            total_length += length
            # Get paths
            paths_cluster = []
            for s_p in solution:
                path_p = []
                for i in range(self.max_steps):
                    id = np.nonzero(s_p[i])[0][0]
                    if i > 0 and id == hub_id:
                        path_p.append(order_to_original[id])
                        break
                    else:
                        path_p.append(order_to_original[id])
                paths_cluster.append(path_p)
            paths.append(paths_cluster)
        return total_length, paths
    
    
    def run_tests(self, clusters, hub_ids, n_runs, aggregation=np.mean, subset_size=None):
        '''
        Generate and run multiple CVRP tasks. Then average the total length \
        of the solution over these runs.

        Params:
        -------
        clusters : list[set[int]]
                   List of sets containing ids of vertices
        hub_ids : list[int]
                  List of ids of vertices, which will serve as hubs
        n_runs : int
                 Number of runs to perform

        Returns:
        --------
        mean : float
               total length mean 
        std : float
              total length std
        '''
        
        # Set seed for reproducibility
        np.random.seed(self.seed)
        scores = []
        # Solve the problem multiple times
        for trial in range(n_runs):
            # Generate demand in each vertex of G
            self.generate_demand()
            # Find solution
            score, path = self.test_cluster(clusters, hub_ids, subset_size)
            scores.append(score)

        return aggregation(scores)
    
    
    def benchmark(self, cluster_alg, cluster_args, hub_strategy=None, n_runs=10, aggregation=np.mean, subset_size=None):
        '''
        Benchmark provided clustering algorithm on CVRP task.
        
        Params:
        -------
        cluster_alg : callable
                      Clustering algorithm, which returns a list of clusters it selected
        cluster_args : dict
                       Dictionary of keywords to pass inside cluster_alg
        n_runs : int
                 Number of runs to perform for benchmark. \
                 The results of these runs will be averged

        Returns:
        --------
        mean : float
               total length mean 
        std : float
              total length std
        '''
        # Find clusters
        clusters = cluster_alg(self.G, **cluster_args)
        # Set the hubs
        if hub_strategy:
            hubs = hub_strategy(self.G, clusters)
        else:
            hubs = []
            for cluster in clusters:
                G_cluster = self.G.subgraph(cluster)
                hubs.append(nx.barycenter(G_cluster, weight=self.weight)[0])   
        # Compute performance
        return self.run_tests(clusters, hubs, n_runs, aggregation, subset_size)


    def plot_routes(self, solution, length, pos=None, show=False):
        '''
        Method for plotting CVRP solutions

        Params:
        -------
        solution : list[list[int]]
                   Paths of CVRP solution
        length : float
                 Total length of the solution
        pos : Any
              Layout for graph vertices
        show : bool
               Execute plt.show() or not
        '''

        K = [
                list(self.G.nodes), # dim = 0 : vertices
                [set(e) for e in self.G.edges], # dim = 1 : edges
            ]
        plot_simplex(pos, K)

        if pos is None:
            pos = nx.spring_layout(self.G, iterations=100)

        for i, s in enumerate(solution):
            color = '#%06X' % randint(0, 0xFFFFFF)
            # vertices
            points = np.array([pos[v] for v in s])
            plt.scatter(points[:, 0], points[:, 1], c=color)
            plt.scatter(points[0, 0], points[0, 1], c='red')

            # plot edges
            for (start_id, end_id) in [(s[i], s[i+1]) for i in range(len(s) - 1)]:
                start_point = pos[start_id]
                end_point = pos[end_id]
                plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], c=color)

        plt.title(f'total length: {length}')
        if show:
            plt.show()


# CVRP Hubs strategies
def border_hubs(clusters, max_size, num_workers=None, time_limit=None, verbose=False):
    # -------------------
    # Part 1: preparation
    # Label all nodes according to their clusters
    clusters_labels = {i : c for i, c in enumerate(clusters)}
    nodes_labels = {}
    for i, c in enumerate(clusters):
        for v in c:
            nodes_labels[v] = i

    # Find border nodes
    border = {}
    for i, cluster in enumerate(clusters):
        for v in cluster:
            # 1-hop neighbour
            for hop_1 ,_ in G._adj[v].items():
                if nodes_labels[hop_1] != i:
                    if v not in border:
                        border[v] = set([nodes_labels[v]])
                    # Store what kind of clusters are nearby
                    border[v].add(nodes_labels[hop_1])
                
                if v in border:
                    # 2-hop neighbour
                    for hop_2 ,_ in G._adj[hop_1].items():
                        if nodes_labels[hop_2] != i:
                            if v not in border:
                                border[v] = set([nodes_labels[v]])
                            # Store what kind of clusters are nearby
                            border[v].add(nodes_labels[hop_2])

    # Filter out points, whose combinations are too large
    erase_list = []
    for v in border:    
        combination_size = 0
        for label in border[v]:
            combination_size += len(clusters[label])
        
        if max_size and combination_size > max_size:
            erase_list.append(v)

    for v in erase_list:
        border.pop(v)

    # Collect combinations of neighbours
    cluster_combinations = defaultdict(lambda: set())
    for v in border:
        combination = tuple(sorted(border[v]))
        cluster_combinations[combination].add(v)

    # Find among them the one, with smallest distance to the whole set
    for combination in cluster_combinations:
        # Get union of all nodes from corresponding clusters
        cluster_union = set()
        for i in combination:
            cluster_union.update(clusters_labels[i])
        G_cluster = nx.subgraph(G, cluster_union)

        if len(cluster_combinations[combination]) > 1:
            # Search for node with minimal distance
            min_dist = np.inf
            for v in cluster_combinations[combination]:
                d = dict(nx.single_source_dijkstra_path_length(G_cluster, v, weight='length'))
                d_sum = np.sum([d[v] for v in d])
                if d_sum < min_dist:
                    d_mean = np.mean([d[v] for v in d])
                    border_hub = v
                    min_dist = d_sum

            cluster_combinations[combination] = (border_hub, d_mean)
        else:
            v = list(cluster_combinations[combination])[0]
            d = dict(nx.single_source_dijkstra_path_length(G_cluster, v, weight='length'))
            d_mean = np.mean([d[v] for v in d])
            cluster_combinations[combination] = (border_hub, d_mean)

    # Now add variants with just centroids
    # centroid_hubs = [None for i in range(len(clusters_labels))]
    for i in clusters_labels:
        G_cluster = G.subgraph(clusters_labels[i])
        hub = nx.barycenter(G_cluster, weight='length')[0]
        d = dict(nx.single_source_dijkstra_path_length(G_cluster, hub, weight='length'))
        d_mean = np.mean([d[v] for v in d])
        # cluster_combinations[i] = (hub, d_mean)
        cluster_combinations[tuple([i])] = (hub, d_mean)

    # --------------------------------
    # Part 2: finding optimal solution
    # Init optimization model
    model = cp_model.CpModel()
    # Store combinations as binary vectors
    Covers = np.zeros((len(cluster_combinations), len(clusters)))
    for i, combination in enumerate(cluster_combinations):
        for j in combination:
            Covers[i][j] = 1
    # Store mean distances as one vector
    Costs = np.array([d[-1] for _, d in cluster_combinations.items()])
    # Some dimension parameters
    N = len(clusters) # len of universe
    P = len(cluster_combinations) # number of sets
    # Variable to store the solution
    Set_cover = {}

    # Each set is a binary vector: [0,0,1,0,1,...]
    for p in range(P):
        Set_cover[p] = model.new_bool_var(name=f'cover_{p}')

    # Sum should give precisely [1,1,1,1,1,...]
    for i in range(N):
        model.add(sum(Set_cover[p] * Covers[p][i] for p in range(P)) == 1)

    # Minimize distance
    objective_func = sum(Set_cover[p] * Costs[p] for p in range(P))
    model.minimize(objective_func)

    solver = cp_model.CpSolver()
    if num_workers:
        solver.parameters.num_workers = num_workers
    solver.parameters.log_search_progress = verbose
    if time_limit:
        solver.parameters.max_time_in_seconds = 60.0 * time_limit # in minutes
    
    status = solver.solve(model)

    status_codes = {
                cp_model.UNKNOWN : 'UNKNOWN',
                cp_model.MODEL_INVALID : 'MODEL_INVALID',
                cp_model.FEASIBLE : 'FEASIBLE',
                cp_model.INFEASIBLE : 'INFEASIBLE',
                cp_model.OPTIMAL : 'OPTIMAL'
            }

    optimal_combination = []
    solver_solution = [solver.value(v) for v in Set_cover.values()]
    for i, cover in enumerate(cluster_combinations):
        if solver_solution[i]:
            optimal_combination.append(cover)

    if verbose:
        print(f'Solution status: {status_codes[status]}')
        print(f'Optimal combination: {optimal_combination}')
        print(f'Total mean distance: {solver.value(objective_func)}')
    
    optimal_hubs = []
    for combination in optimal_combination:
        optimal_hubs.append(cluster_combinations[combination][0])
    
    return optimal_hubs