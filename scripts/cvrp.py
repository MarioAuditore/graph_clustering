import networkx as nx
import numpy as np

from .graph_filtration.utils import plot_simplex

from ortools.sat.python import cp_model
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from random import randint


class TestCVRP:
    def __init__(self, G, max_steps, n_vehicles, weight='length', n_repeats = 10, time_limit = None, seed=0xAB0BA):
        self.G = G
        self.weight = weight
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.n_repeats = n_repeats
        self.seed = seed
        self.P = n_vehicles
    
    def find_solution(self, d_matrix, capacities, demands, hub_id = 0):
        # Define constraint model
        model = cp_model.CpModel()

        # defining variables
        Routes = {}  # x_ijp  - матрица маршрутов
        Y = {} # произведения ij

        # d_matrix = self.d_matrix
        MAX_STEPS = self.max_steps
        N = len(d_matrix)
        P = len(capacities)
        # capacities = self.capacities
        # demands = self.demands
        # hub_id = self.hub_id
        #____________________
        # Создание переменных
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
                            model.add_multiplication_equality(Y[p, r, i, j], [Routes[p, r, i], Routes[p, r + 1, j]])
                        else:
                            model.add(Y[p, r, i, j] == 0)
        
        #_____________________
        # Ограничения на спрос
        for p in range(P):
            model.add(sum(Routes[p, r, i] * demands[i] for i in range(N) for r in range(1, MAX_STEPS - 1)) <= capacities[p])

        #_________________________________
        # ограничение из матрицы смежности
        for i in range(N):
            for j in range(N):
                for r in range(MAX_STEPS - 1):
                    for p in range(P):
                        if d_matrix[i, j] == 0 and not (i == hub_id and j == hub_id):
                            model.add(Routes[p, r + 1, j] <= 1 - Routes[p, r, i])

        # x_iip = 0 - не ездить из города в себя
        for i in range(N):
            if i != hub_id:
                for r in range(MAX_STEPS - 1):
                    for p in range(P):
                        model.add(Routes[p, r + 1, i] <= 1 - Routes[p, r, i])

        for p in range(P):
            # изначально в хабе
            model.add(Routes[p, 0, hub_id] == 1)
            # # на первом шаге выезжаем из него
            # model.add(Routes[p, 1, hub_id] <= 1 - Routes[p, 0, hub_id])

            for r in range(MAX_STEPS):
                model.add(sum(Routes[p, r, i] for i in range(N)) == 1)  # на каржом роутсе только в одном месте

            # если на r-м шаге не оказались, то и дальше не едем; если на r-м шаге в хабе, то дальше не едем; если на r-м шаге оказались не в хабе, то едем дальше 
            for r in range(1, MAX_STEPS - 1):
                model.add(Routes[p, r + 1, hub_id] >= Routes[p, r, hub_id])  #если приехали в хаб то остаемся в нем
            # возвращаемся в хаб
            model.add(sum(Routes[p, r, hub_id] for r in range(1, MAX_STEPS)) >= 1)

        # Ensure that every node is entered at least once
        for i in range(N):
            if i != hub_id:
                model.add(sum(Routes[p, r, i] for p in range(P) for r in range(1, MAX_STEPS)) == 1)
                for p in range(P):
                    model.add(sum(Routes[p, r, i] for r in range(1, MAX_STEPS)) <= 1)

        # Minimize distance
        objective_func = sum(Y[p, r, i, j] * d_matrix[i,j] for r in range(MAX_STEPS - 1) for i in range(N) for j in range(N) for p in range(P))
        model.minimize(objective_func)
        
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.max_time_in_seconds = 60.0 * self.time_limit # in minutes
        # callback = Callback(Routes, T_out, Car_Type, day, solver, objective_func)

        # solution_collector = VarArraySolutionCollector(Routes)

        status = solver.solve(model)
        print(status)
        if status == 3:
            return None

        return np.reshape([solver.value(v) for v in Routes.values()], (P, MAX_STEPS , N)), solver.value(objective_func)
    

    def test_cluster(self, clusters=None, hub_ids=None):
        def get_dist_matrix(G, weight='length'):
            # G = nx.convert_node_labels_to_integers(G)
            N = len(G.nodes)
            d_matrix = np.zeros((N, N))
            lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
            # re-numerate
            original_to_order = {}
            order_to_original = {}
            for i, id in enumerate(lengths):
                original_to_order[id] = i
                order_to_original[i] = id

            for i in lengths:
                for j in lengths[i]:
                    d_matrix[original_to_order[i], original_to_order[j]] = lengths[i][j]
            return d_matrix, order_to_original, original_to_order
        
        # Set seed
        np.random.seed(self.seed)
        # number of vehicles
        P = self.P

        if clusters is None:
            clusters = [list(self.G.nodes)]
            hub_ids = [0]
        
        paths = []
        total_length = 0
        
        for i, cluster in enumerate(clusters):
            # Get graph of cluster
            G_cluster = self.G.subgraph(cluster)
            # Get it's pairwise distance
            d_matrix, order_to_original, original_to_order = get_dist_matrix(G_cluster, self.weight)
            # Choose hub
            hub_id = original_to_order[hub_ids[i]]
            # Generate capacities and demands
            N = len(cluster)
            capacities = np.ones(P, dtype=int)
            # demands = np.ones(N, dtype=int) * 100
            demands = np.array([data[-1]['demand'] for data in G_cluster.nodes(data=True)])
            demands[hub_id] = 0
            while demands.sum() > capacities.sum():
                capacities += np.random.randint(0, 5, size=P)
                # demands = np.random.randint(1, 5, size=N)
                # demands[hub_id] = 0
            
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
    
    
    def run_tests(self, clusters, hub_ids):
        # Set seed for reproducibility
        np.random.seed(self.seed)
        scores = []
        # Solve the problem multiple times
        for trial in tqdm(range(self.n_repeats)):
            # Generate new demands
            demands = {node: np.random.randint(1, 5, 1)[0] for node in self.G.nodes}
            nx.set_node_attributes(self.G, demands, 'demand')
            # Find solution
            total_length, _ = self.test_cluster(clusters, hub_ids)
            scores.append(total_length)

        return np.mean(scores), np.std(scores)
    
    
    def benchmark(self, cluster_alg, cluster_args):
        # Find clusters
        clusters = cluster_alg(self.G, **cluster_args)
        # Set the hubs
        hubs = []
        for cluster in clusters:
            G_cluster = self.G.subgraph(cluster)
            hubs.append(nx.barycenter(G_cluster, weight=self.weight)[0])   
        # Compute performance
        return self.run_tests(clusters, hubs)


    def plot_routes(self, solution, length, pos=None, show=False):

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