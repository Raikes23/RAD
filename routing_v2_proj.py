# -*- coding: utf-8 -*-
"""
@authors: Jose Ribeiro (Graph class, getPaths, shortestPaths, countHops and create_traffic_matrix)
          Alexandre Freitas (orderPaths, create_load_matrix, update_network, breakTie, route, route_path, hop_count and printResults)
"""

import numpy as np
import copy
import itertools
import time
import matplotlib.pyplot as plt
import os
from itertools import combinations

import networkx as nx
print("NetworkX version: {}".format(nx.__version__))


# Class to represent a graph
class Graph:

    # A utility function to find the
    # vertex with minimum dist value, from
    # the set of vertices still in queue
    def minDistance(self, dist, queue):
        # Initialize min value and min_index as -1
        minimum = float("Inf")
        min_index = -1

        # from the dist array,pick one which
        # has min value and is till in queue
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index

    # Function to print shortest path
    # from source to j
    # using parent array
    def printPath(self, parent, j):
        path = []
        # Base Case : If j is source
        if parent[j] == -1:
            # print (j+1)
            return [j + 1]

        path.extend(self.printPath(parent, parent[j]))
        # print (j+1)
        path.append(j + 1)
        return path

        # A utility function to print

    # the constructed distance
    # array
    def printSolution(self, src, dist, parent):
        paths = []
        # print("Vertex \t\tDistance from Source\tPath")
        for i in range(0, len(dist)):
            # print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src+1, i+1, dist[i])),
            path = self.printPath(parent, i)
            paths.append({
                "source": src + 1,
                "destination": i + 1,
                "distance": dist[i],
                "path": path
            })
        return paths

    '''Function that implements Dijkstra's single source shortest path
    algorithm for a graph represented using adjacency matrix
    representation'''

    def dijkstra(self, graph, src):

        row = len(graph)
        col = len(graph[0])

        # The output array. dist[i] will hold
        # the shortest distance from src to i
        # Initialize all distances as INFINITE
        dist = [float("Inf")] * row

        # Parent array to store
        # shortest path tree
        parent = [-1] * row

        # Distance of source vertex
        # from itself is always 0
        dist[src] = 0

        # Add all vertices in queue
        queue = []
        for i in range(row):
            queue.append(i)

        # Find shortest path for all vertices
        while queue:

            # Pick the minimum dist vertex
            # from the set of vertices
            # still in queue
            u = self.minDistance(dist, queue)

            # remove min element
            if u != -1:
                queue.remove(u)

                # Update dist value and parent
                # index of the adjacent vertices of
                # the picked vertex. Consider only
                # those vertices which are still in
                # queue
                for i in range(col):
                    '''Update dist[i] only if it is in queue, there is
                    an edge from u to i, and total weight of path from
                    src to i through u is smaller than current value of
                    dist[i]'''
                    if graph[u][i] and i in queue:
                        if dist[u] + graph[u][i] < dist[i]:
                            dist[i] = dist[u] + graph[u][i]
                            parent[i] = u
            else:
                queue.clear()

        # print the constructed distance array
        return self.printSolution(src, dist, parent)

# New functions for average node degree, variance, and distribution

def calculate_average_node_degree(adjacency_matrix):
    degrees = np.sum(adjacency_matrix, axis=1)
    average_degree = np.mean(degrees)
    return average_degree, degrees

def calculate_degree_variance(degrees):
    variance = np.var(degrees)
    return variance

def plot_degree_distribution(degrees):
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), edgecolor='black', align='left')
    plt.title('Node Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def ensure_unweighted(adjacency_matrix):
    return (adjacency_matrix > 0).astype(int)

def getPaths(graph: Graph, matrix: list):
    patths = []
    # Print the solution
    for i in range(len(matrix)):
        patths.append(graph.dijkstra(matrix, i))

    return patths

def shortestPaths(graph: Graph, matrix: list):
    pairs = []
    paths = []

    #count = min([len([i for i in row if i > 0]) for row in matrix])
    count = 2
    for i in range(0, len(matrix)):
        for j in range(i, len(matrix)):
            if matrix[i][j] != 0:
                pairs.append(f'{i + 1}-{j + 1}')

    for i in range(len(matrix)):
        if i > count:
            break
        combinationss = list(itertools.combinations(pairs, i))

        for comb in combinationss:
            aux_matrix = copy.deepcopy(matrix)

            for pair in comb:
                row, column = pair.split('-')
                aux_matrix[int(row) - 1][int(column) - 1] = 0
                aux_matrix[int(column) - 1][int(row) - 1] = 0

            # print(f'Pair removed: {comb}')
            # for row in range(len(np.array(aux_matrix))):
            # print(aux_matrix[row])

            if not (~np.array(aux_matrix).any(axis=0)).any():
                aux_paths = getPaths(graph, aux_matrix)
                if len(paths) == 0:
                    paths = aux_paths
                    for path in paths:
                        for p in path:
                            p["path"] = [p["path"]]
                else:
                    for a, path in enumerate(paths):
                        for b, p in enumerate(path):
                            if aux_paths[a][b]["distance"] == p["distance"] and aux_paths[a][b]["path"] not in p[
                                    "path"]:
                                p["path"].append(aux_paths[a][b]["path"])
    return paths

# Further operations remain the same

def getPaths(graph: Graph, matrix: list):
    patths = []
    # Print the solution
    for i in range(len(matrix)):
        patths.append(graph.dijkstra(matrix, i))

    return patths


########################################################################################################
#                                  Função modificada
########################################################################################################
def countHops(paths: list): #return hop matrix
    hop_matrix = np.zeros((len(paths), len(paths)))
    for i in range(len(paths)):
        for j in range(len(paths)):
            if i == j:
                continue
            for path in paths[i]:
                if path["destination"] == j + 1:
                    hop_matrix[i][j] = len(path["path"]) - 1
                    break
    shape = hop_matrix.shape
    print("shape",shape)   
    return hop_matrix




def create_traffic_matrix(matrix, traffic):
    matrix_size = len(matrix)
    if traffic == None or len(traffic) != matrix_size:
        a = np.ones((matrix_size, matrix_size), int)
        np.fill_diagonal(a, 0)
        return a.tolist()
    else:
        return traffic

# Returns the ordered traffic demands according to a given sorting strategy 
# (shortest, longest or largest) based on the distance.     
def orderPaths(paths: list, traffic_matrix: list, hop_matrix: list, order = "shortest"):
    
    path_list = []
    # Iterate over the shortest-paths list.
    for path in paths:
        for p in path:
            p["traffic"] = traffic_matrix[p["source"]-1][p["destination"]-1]
            p["hops"] = hop_matrix[p["source"]-1][p["destination"]-1]
            p['routed'] = False
            
            # If there are multiple shortest-paths, remove the ones with larger
            # number of hops.
            possible_paths = p["path"]
            if len(possible_paths) > 1:
                length_min_path = min(len(x) for x in possible_paths)
                possible_paths_aux = possible_paths.copy()
                for idx, item in enumerate(possible_paths):
                    if len(item) > length_min_path:
                        possible_paths_aux.remove(item)
                        p['path'] = possible_paths_aux
            
            # Include only traffic demands with source different than destination
            # and with traffic.
            if p["source"] != p["destination"] and p["traffic"] > 0:
                path_list.append(p)
    
    # Order the paths.    
    if order == "shortest":
        ordered_paths = sorted( path_list, key = lambda d: (d['distance'], len(d['path'])) )
    elif order == "longest":
        ordered_paths = sorted( path_list, key = lambda d: (d['distance'], -len(d['path'])),reverse = True )
    elif order == "largest":
        ordered_paths = sorted( path_list, key = lambda d: (d['traffic'], -len(d['path'])),reverse = True )
    else:
        ordered_paths = sorted( path_list, key = lambda d: (d['distance'], len(d['path'])) )

    return ordered_paths

# Create NxN matrix to represent the load of each one of the links in the network
def create_load_matrix(matrix):
    matrix_size = len(matrix)
    a = np.zeros((matrix_size, matrix_size), int)
    return a.tolist()

# Updates the network by removing links that have become saturated with traffic and then
# recalculates the shortest-paths. Called on route function.
def update_network(links_to_remove,matrix):
    # Iterate over each link to be removed and set the length 
    # of the link in the network to a large value (virtually removes it).
    for link in links_to_remove:
        matrix[link[0]-1][link[1]-1] = 99999
    
    # Recalculate the shortest paths.
    new_shortest_paths = shortestPaths(graph,matrix)

    # If there are multiple shortest paths, remove the ones with larger
    # number of hops.
    for path in new_shortest_paths:
        for p in path:            
            possible_paths = p["path"]
            if len(possible_paths) > 1:
                length_min_path = min(len(x) for x in possible_paths)
                possible_paths_aux = possible_paths.copy()
                for idx, item in enumerate(possible_paths):
                    if len(item) > length_min_path:
                        possible_paths_aux.remove(item)
                        p['path'] = possible_paths_aux
                
    return new_shortest_paths

# Function that, if there are multiple paths to the destination, returns the one 
# that minimizes the maximum load between the links.
def breakTie(p:dict,load_matrix:list):
    loads_of_path = []
    all_loads = [None] * len (p['path'])
    # Determines loads for each path.
    for idx,_ in enumerate(p['path']):
        for x,y in zip(p['path'][idx],p['path'][idx][1:]):
            loads_of_path.append(load_matrix[x-1][y-1])
            
        all_loads[idx]= copy.deepcopy(loads_of_path)
        loads_of_path.clear()
        # all_loads is a list of lists where each inner list has the loads in each link of a path.

    maximum_list = []
    # After having all the loads in all the paths, find the path with the minimum maximum load.
    while True:
        for path_load in all_loads:
            # If all loads in all paths are the same in the end, just pick the first path in the list (index 0).
            if len(path_load) == 0:
                chosen_path = 0
                return chosen_path  
            
            maximum_list.append(max(path_load))
            path_load.remove(max(path_load))
        
        # If all elements in maximum_list are the same (there's no minimum), 
        # move on to the links with next largest loads.
        if all(ele == maximum_list[0] for ele in maximum_list):
            maximum_list.clear()
            continue
        # Not all equal, there's a minimum:
        else:
            # Check if min is unique. If so, the path was found, else go to next largest loads.
            min_val = min(maximum_list)
            if maximum_list.count(min_val) == 1:
                chosen_path = maximum_list.index(min_val)
                return chosen_path
            else:
                maximum_list.clear()
                continue

# Auxiliary function to the route function, updates the matrices (load_matrix, path_matrix) 
# and checks if there are links that need to be removed (adds them to a list) and sets
# update_net_flag to True in case a link has residual capacity zero.
def route_path(p, load_matrix, path_matrix, chosen_path, p_aux, links_to_remove):
    update_net_flag = False
    blocked_flag = False
    
    # Case where traffic has value greater than one.
    if p_aux['traffic'] > 1:
        # Check all links to see if traffic fits the entire path, if not set blocked_flag as True.
        for x,y in zip(p['path'][chosen_path],p['path'][chosen_path][1:]):
            if load_matrix[x-1][y-1] + p_aux['traffic'] > MAX_LINK_CAP:
                blocked_flag = True
        
        # Case the traffic does not fit the path (is blocked):
        if blocked_flag:
            return update_net_flag, blocked_flag
    
    # Update the matrices accordingly.
    path_matrix[p['source']-1][p['destination']-1] = p['path'][chosen_path]
    # Iterate over all edges of the chosen path:
    for x,y in zip(p['path'][chosen_path],p['path'][chosen_path][1:]):
        load_matrix[x-1][y-1] += p_aux['traffic']
        # Case of link saturation
        if load_matrix[x-1][y-1] == MAX_LINK_CAP:
            update_net_flag = True
            links_to_remove.append((x,y))

    return update_net_flag, blocked_flag

# Routes the traffic according to ordered_traffic_demands. Returns the completed load_matrix, path_matrix,
# distance_matrix, blocked_traffic (number of blocked paths), and blocked_paths (list of blocked paths).
def route(ordered_traffic_demands, load_matrix:list,matrix):
    # Initialize matrices and lists
    a = np.zeros((len(load_matrix),len(load_matrix)),int)
    path_matrix = a.tolist()
    distance_matrix = a.tolist()
    
    new_paths_flag = False
    blocked_traffic = 0
    blocked_paths = []
    links_to_remove = []
    
    # Iterate on the ordered list of traffic demands.
    for p in ordered_traffic_demands:
        if p['routed'] == True:
            continue
        
        p['routed'] = True
        
        links_to_remove = []
        
        # p_aux is the p in the original ordered_traffic_demands, 
        # because the recalculated paths do not have 'traffic' in dictionary.
        p_aux = p
        
        # In case there was a recalculation of shortest-paths in every iteration p
        # is now from the new_shortest_paths list.
        if new_paths_flag:
            p = new_shortest_paths[p['source']-1][p['destination']-1]

        distance_matrix[p['source']-1][p['destination']-1] = p['distance']
        
        # In case a path does not exist (path_matrix value will be kept at 0)
        if p['distance'] >= 99999:
            blocked_traffic += p_aux['traffic']
            blocked_paths.append(p)
            continue
        
        # Choose the path index of the path to route through.
        # Only a single path between source and destination.
        if len(p['path']) == 1:
            chosen_path = 0
        # In case there is a path tie (two or more paths between source and destination)
        else:
            chosen_path = breakTie(p,load_matrix)

        update_net_flag, blocked_flag = route_path(p, load_matrix, path_matrix, chosen_path, p_aux,links_to_remove)
        # In case traffic > 1, blocked traffic can occur inside each iteration.
        if blocked_flag:
            distance_matrix[p['source']-1][p['destination']-1] = 0
            blocked_traffic += p_aux['traffic']
            blocked_paths.append(p)
            continue
        
        # So that symmetric connections are made one after the other and through the same path.
        for p2 in ordered_traffic_demands:
            if p2['source'] == p['destination'] and p2['destination'] == p['source'] and p_aux['traffic'] == p2['traffic'] and p2['routed'] == False:
                
                p2['routed'] = True

                if new_paths_flag:
                    p2 = new_shortest_paths[p2['source']-1][p2['destination']-1]

                distance_matrix[p2['source']-1][p2['destination']-1] = p2['distance']
                
                # Choose same path:
                chosen_path2 = 0
                while True:    
                    # check if path of p contains all elements of current path of p2
                    result = all(elem in p['path'][chosen_path] for elem in p2['path'][chosen_path2])
                    if result == True:
                        break
                    else:
                        chosen_path2 += 1

                update_net_flag, _ = route_path(p2, load_matrix, path_matrix, chosen_path2, p_aux, links_to_remove)
                
                break
                
        # If there is a link that has reached the limit capacity, network must be updated 
        # (remove the links that are saturated from matrix and find new shortest-paths).
        if update_net_flag:
            new_shortest_paths = update_network(links_to_remove,matrix)
            new_paths_flag = True
            
    return load_matrix, path_matrix, distance_matrix, blocked_traffic, blocked_paths

# Prints the ordered traffic demands and loads in every link and calculates average load per link.
def printResults (ordered_paths:list, adj_matrix:list,load_matrix: list,distance_matrix:list):
    print("Ordered Paths:")
    for p in ordered_paths:
        if p['routed'] == True:
            if path_matrix[p['source']-1][p['destination']-1] != 0:
                print("Path: {} , Dist: {} , Og.Dist: {} , traffic = {} (Og. Path: {})".format(path_matrix[p['source']-1][p['destination']-1],
                                                                                                       distance_matrix[p['source']-1][p['destination']-1],
                                                                                                       p['distance'], traffic[p['source']-1][p['destination']-1],p['path']))
            elif path_matrix[p['source']-1][p['destination']-1] == 0:
                print("Path: BLOCKED , Dist: BLOCKED , Og.Dist: {} , traffic = {} (Og. Path: {})".format(p['distance'],traffic[p['source']-1][p['destination']-1],p['path']))
            p['routed'] = False
        for p2 in ordered_paths:
            if p2['source'] == p['destination'] and p2['destination'] == p['source'] and p['traffic'] == p2['traffic'] and p2['routed'] == True:
                if path_matrix[p2['source']-1][p2['destination']-1] != 0:
                    print("Path: {} , Dist: {} , Og.Dist: {} , traffic = {} (Og. Path: {})".format(path_matrix[p2['source']-1][p2['destination']-1],
                                                                                                     distance_matrix[p2['source']-1][p2['destination']-1], p2['distance'],
                                                                                                     traffic[p2['source']-1][p2['destination']-1],p2['path']))
                elif path_matrix[p2['source']-1][p2['destination']-1] == 0:
                    print("Path: BLOCKED , Dist: BLOCKED , Og.Dist: {} , traffic = {} (Og. Path: {})".format(p2['distance'],traffic[p2['source']-1][p2['destination']-1],p2['path']))
                #print("Path: {} -> traffic = {}".format(path_matrix[p2['source']-1][p2['destination']-1],traffic[p2['source']-1][p2['destination']-1]))
                p2['routed'] = False 
   
    accum_sum = 0
    n_links = 0
    print("---LOADS IN EVERY LINK---")
    for row,_ in enumerate(load_matrix):
        for collumn,_ in enumerate (load_matrix[row]):
            if adj_matrix[row][collumn] == 0:
                continue
            print("{}-{} -> {}".format(row+1,collumn+1,load_matrix[row][collumn]))
            #print("{}->{}".format(row+1,collumn+1))
            #print(load_matrix[row][collumn])
            accum_sum += load_matrix[row][collumn]
            n_links += 1
    
    average_load_per_link = accum_sum/n_links
    return average_load_per_link

# Returns hop_matrix and avg_hops_per_path calculated through the path_matrix, 
# (the returned values refer to the final chosen paths after routing).
def hop_count(path_matrix):
    hop_matrix = np.zeros((len(path_matrix),len(path_matrix)))
    for row,_ in enumerate(path_matrix):
        for collumn,_ in enumerate (path_matrix[row]):
            if path_matrix[row][collumn] == 0:
                continue

            hop_matrix[row][collumn] = len(path_matrix[row][collumn]) - 1
    
    avg_hops_per_path = np.mean(hop_matrix[np.nonzero(hop_matrix)])
    
    return hop_matrix,avg_hops_per_path

# Function to check if matrix is symmetric 
# (with a tolerance due to the limitations of floating point precision)  
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

# Function to check if all elements on main diagonal are zero. Return True if so.
def check_diagonal_zero(matrix):
    diagonal = np.diagonal(matrix)
    return np.all(diagonal == 0)

#####################################################################################################
#                               Funções para alinea 4 da Segunda fase
#####################################################################################################

# Função para verificar componentes conectados
def count_connected_components(matrix):
    visited = set()

    def dfs(node, visited):
        visited.add(node)
        for neighbor, connected in enumerate(matrix[node]):
            if connected and neighbor not in visited:
                dfs(neighbor, visited)

    components = 0
    for node in range(len(matrix)):
        if node not in visited:
            dfs(node, visited)
            components += 1
    return components

# Node Connectivity
""" def calculate_node_connectivity(matrix):
    original_components = count_connected_components(matrix)
    min_cut = len(matrix)  # Initialize with the total number of nodes (worst case).

    for node in range(len(matrix)):
        # Remove the current node (and its edges) to simulate its removal.
        reduced_matrix = np.delete(np.delete(matrix, node, axis=0), node, axis=1)
        components = count_connected_components(reduced_matrix)
        
        # If the graph becomes disconnected, count this node as part of the cut set.
        if components > original_components:
            min_cut = min(min_cut, 1)  # Removing a single node can disconnect the graph.
    
    return min_cut """

# Edge Connectivity
""" 
def calculate_edge_connectivity(matrix):
    original_components = count_connected_components(matrix)
    edge_connectivity = np.inf

    edges = [(i, j) for i in range(len(matrix)) for j in range(i + 1, len(matrix)) if matrix[i][j]]
    for edge_set in combinations(edges, 1):  # 1 aresta por vez
        test_matrix = matrix.copy()
        for i, j in edge_set:
            test_matrix[i][j] = test_matrix[j][i] = 0
        components = count_connected_components(test_matrix)
        if components > original_components:
            edge_connectivity = min(edge_connectivity, len(edge_set))
    return edge_connectivity """

def calculate_node_connectivity(matrix):
    """
    Calculate the node connectivity of a graph represented by an adjacency matrix.
    """
    G = nx.from_numpy_array(np.array(matrix))  # Use from_numpy_array instead
    return nx.node_connectivity(G)

def calculate_edge_connectivity(matrix):
    """
    Calculate the edge connectivity of a graph represented by an adjacency matrix.
    """
    G = nx.from_numpy_array(np.array(matrix))  # Use from_numpy_array instead
    return nx.edge_connectivity(G)

def calculate_average_degree(matrix):
    """
    Calculate the average node degree of a graph represented by an adjacency matrix.
    """
    degrees = [sum(row) for row in matrix]  # Sum each row for the degree of each node
    return sum(degrees) / len(degrees)

#####################################################################################################
#                               Funções multiplica fator de distancia
#####################################################################################################
def multiply_matrix_by_0_95(matrix):
    # Converte a lista de listas em um array numpy para manipulação eficiente
    np_matrix = np.array(matrix)
    
    # Multiplica todos os elementos por 0.95
    np_matrix = np_matrix * 0.95
    
    # Retorna a matriz resultante
    return np_matrix

#####################################################################################################
#                               Funções para alinea 5 da Segunda fase
#####################################################################################################
# Função para verificar se x e y estão conectados
def are_connected(matrix, x, y):
    visited = set()

    def dfs(node):
        if node == y:
            return True
        visited.add(node)
        for neighbor, connected in enumerate(matrix[node]):
            if connected and neighbor not in visited:
                if dfs(neighbor):
                    return True
        return False

    return dfs(x)

""" # Determinar o corte mínimo de nós
def find_minimum_node_cut(matrix, x, y):
    print("Matrix: ", matrix)
    original_matrix = matrix.copy()
    node_cut_set = []

    for node in range(len(matrix)):
        if node == x or node == y:
            continue
        reduced_matrix = np.delete(np.delete(matrix, node, axis=0), node, axis=1)
        if not are_connected(reduced_matrix, x, y):
            node_cut_set.append(node)

    matrix[:] = original_matrix  # Restaurar a matriz original
    return node_cut_set """

""" # Determinar o corte mínimo de arestas
def find_minimum_edge_cut(matrix, x, y):
    original_matrix = matrix.copy()
    edge_cut_set = []

    edges = [(i, j) for i in range(len(matrix)) for j in range(i + 1, len(matrix)) if matrix[i][j]]
    for i, j in edges:
        test_matrix = matrix.copy()
        test_matrix[i][j] = test_matrix[j][i] = 0
        if not are_connected(test_matrix, x, y):
            edge_cut_set.append((i, j))

    matrix[:] = original_matrix  # Restaurar a matriz original
    return edge_cut_set """

def find_minimum_node_cut(matrix, x, y):
    """
    Find the minimum node cut set between nodes x and y in a graph represented by an adjacency matrix.
    """
    # Create the graph from the adjacency matrix
    G = nx.from_numpy_array(np.array(matrix))
    
    # Find and return the minimum node cut between x and y
    node_cut = nx.minimum_node_cut(G, s=x, t=y)

    # Convert to one-based indexing for output
    return {node + 1 for node in node_cut}

def find_minimum_edge_cut(matrix, x, y):
    """
    Find the minimum edge cut set between nodes x and y in a graph represented by an adjacency matrix.
    """
    # Create the graph from the adjacency matrix
    G = nx.from_numpy_array(np.array(matrix))
    
    # Find and return the minimum edge cut between x and y
    edge_cut = nx.minimum_edge_cut(G, s=x, t=y)

    # Convert to one-based indexing for output
    return {(u + 1, v + 1) for u, v in edge_cut}

#####################################################################################################
#                               Funções para alinea 6 da Segunda fase
#####################################################################################################
# Encontrar o caminho de serviço e caminhos de backup entre x e y
def find_service_and_backup_paths(paths, x, y):
    service_path = None
    backup_paths = []

    # Filtrar os caminhos para x -> y
    xy_paths = [p for p in paths[x] if p["destination"] == y + 1]

    # Encontrar o caminho de serviço (menor distância)
    if xy_paths:
        service_path = min(xy_paths, key=lambda p: p["distance"])

    # Encontrar os caminhos de backup
    for path in xy_paths:
        if path != service_path:
            backup_paths.append(path)

    return service_path, backup_paths

def find_service_and_backup_paths(paths, x, y):
    """
    Find the service path and backup paths between two nodes in the network.

    Parameters:
    paths (list): A list of shortest paths between nodes.
    x (int): Index of the source node.
    y (int): Index of the destination node.

    Returns:
    tuple: A tuple containing the service path and a list of backup paths.
    """
    # Initialize the service path and backup paths
    service_path = None
    backup_paths = []

    x = x +1
    y = y +1

    # Convert the list of paths to a graph representation for easier pathfinding
    graph = nx.Graph()
    for path_set in paths:
        for path_info in path_set:
            path = path_info["path"]
            distance = path_info["distance"]
            for i in range(len(path) - 1):
                graph.add_edge(path[i], path[i + 1], weight=distance)

    try:
        # Get the shortest path (service path) between x and y
        #print the graph
        print(graph)
        
        service_path = nx.shortest_path(graph, source=x, target=y, weight='weight')

        # Remove the service path edges to find backup paths
        graph.remove_edges_from(zip(service_path[:-1], service_path[1:]))

        # Find alternative paths (backup paths)
        for _ in range(3):  # Limit to 3 backup paths for simplicity
            try:
                backup_path = nx.shortest_path(graph, source=x, target=y, weight='weight')
                backup_paths.append(backup_path)
                graph.remove_edges_from(zip(backup_path[:-1], backup_path[1:]))
            except nx.NetworkXNoPath:
                break
    except nx.NetworkXNoPath:
        pass

    return service_path, backup_paths
#####################################################################################################

graph = Graph()
average_node_degree = []
traffic = []
number_of_hops_per_demand = []
diameter = []

# Default value for the link capacity (999999 corresponds to uncapacitated routing)
MAX_LINK_CAP = 999999
#MAX_LINK_CAP = 64
#MAX_LINK_CAP = 75
#------------------------------------INPUT MATRICES HERE------------------------------------------------
# If traffic matrix is not defined, the routing will be done using a full-mesh logical topology with
# one unit of traffic.
#TEST NETWORK
MAX_LINK_CAP = 5

# Criar matriz de tráfego baseada nos valores X, Y, Z
traffic = [
    [0, 29, 46, 55, 29, 46, 55, 29, 46, 55, 29, 46, 55, 29],
    [29, 0, 29, 46, 55, 29, 46, 55, 29, 46, 55, 29, 46, 55],
    [46, 29, 0, 29, 46, 0, 29, 0, 55, 29, 0, 55, 0, 0],
    [55, 46, 29, 0,	29,	46,	55,	0,	0,	55,	29,	0,	55,	0],
    [29, 0, 46,	29,	0,	29,	46,	55,	0, 46,	0,	0, 0, 55],
    [46, 29, 0, 46,	29,	0,	29,	46,	55,	29,	46,	0,	29,	0],
    [0,	0,	29,	55,	46,	29,	0,	29,	55,	29,	46,	0,	55,	29],
    [29,	55,	0,	0,	55, 46,	29,	0,	29,	55,	29,	0,	55,	0],
    [46,	29,	55,	0,	0,	55,	46,	29,	0,	29,	55, 29,	0,	46],
    [0,	46,	29,	55,	46,	29, 55,	46,	29,	0,	29,	55,	46,	29],
    [29,	55,	0,	29,	0,	46,	29,	55,	46,	29,	0,	29,	55,	0],
    [	0,	0,	55,	0,	0,	0,	0,	29,	55,	46,	29,	0,	29,	46],
    [0,	0,	0,	55,	0,	29,	55,	0,	29,	55,	46,	0,	29,	0],
    [0,	55,	0,	0,	55,	0,	29,	55,	46,	29,	55,	46,	46,	0]
]


#####################################################################################################
#                               Primeira fase
#####################################################################################################

#Nodes: 14
#Links: 21
unweighted_matrix = [
    [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Node 1: Seattle
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 2: Palo Alto
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 3: San Diego
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Node 4: Salt Lake City
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Node 5: Boulder
    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # Node 6: Houston
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Node 7: Lincoln
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # Node 8: Champaign
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # Node 9: Atlanta
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # Node 10: Ann Arbor
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],  # Node 11: Pittsburgh
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # Node 12: College Park
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],  # Node 13: Ithaca
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],  # Node 14: Princeton
]   

matrix_weighted = [
    [0, 1130.3, 1700.0, 0, 0, 0, 0, 2829.11, 0, 0, 0, 0, 0, 0],     # Node 1: Seattle
    [1130.3, 0, 693.67, 956.43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # Node 2: Palo Alto
    [1700.0, 693.67, 0, 0, 0, 2093.0, 0, 0, 0, 0, 0, 0, 0, 0],      # Node 3: San Diego
    [0, 956.43, 0, 0, 568.76, 0, 0, 0, 0, 2338.42, 0, 0, 0, 0],     # Node 4: Salt Lake City
    [0, 0, 0, 568.76, 0, 1452.8, 730.16, 0, 0, 0, 0, 0, 0, 0],      # Node 5: Boulder
    [0, 0, 2093.0, 0, 1452.8, 0, 0, 0, 1128.27, 0, 0, 0, 0, 0],     # Node 6: Houston
    [0, 0, 0, 0, 730.16, 0, 0, 719.24, 0, 0, 0, 0, 0, 0],           # Node 7: Lincoln
    [2829.11, 0, 0, 0, 0, 0, 719.24, 0, 0, 0, 700.78, 0, 0, 0],     # Node 8: Champaign
    [0, 0, 0, 0, 0, 1128.27, 0, 0, 0, 0, 839.58, 0, 0, 0],          # Node 9: Atlanta
    [0, 0, 0, 2338.42, 0, 0, 0, 0, 0, 0, 0, 0, 595.04, 788.13],     # Node 10: Ann Arbor
    [0, 0, 0, 0, 0, 0, 0, 700.78, 839.58, 0, 0, 0, 366.69, 452.18], # Node 11: Pittsburgh
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 385.84, 246.96],           # Node 12: College Park
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 595.04, 366.69, 385.84, 0, 0],      # Node 13: Ithaca
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 788.13, 452.18, 246.96, 0, 0]       # Node 14: Princeton
]

matrix_weighted = multiply_matrix_by_0_95(matrix_weighted)

# Ensure matrix is unweighted
matrix = ensure_unweighted(np.array(unweighted_matrix))
print(matrix)

# Calculate average degree, variance, and plot
average_degree, degrees = calculate_average_node_degree(matrix)
variance = calculate_degree_variance(degrees)

print(f"Average Node Degree: {average_degree}")
print(f"Variance of Node Degrees: {variance}")

plot_degree_distribution(degrees)

#########################################################################################################
#                            Primiera alinea da Segunda fase
#########################################################################################################
# Calcular os caminhos mais curtos no grafo ponderado
print("\n")
print("Calculando caminhos mais curtos no grafo ponderado:")
weighted_paths = getPaths(graph, matrix_weighted)
for path in weighted_paths:
    #print(path)
    print("\n")

# Calcular os caminhos mais curtos no grafo não ponderado
print("\nCalculando caminhos mais curtos no grafo não ponderado:")
unweighted_paths = getPaths(graph, matrix)
for path in unweighted_paths:
    #print(path)
    print("\n")

#########################################################################################################
#                             Segunda alinea da Segunda fase
#########################################################################################################

# Extração de hops e distâncias do grafo ponderado
weighted_hops = []
weighted_distances = []
for path_set in weighted_paths:
    for path in path_set:
        if path["distance"] < float("Inf"):  # Ignorar caminhos desconectados
            weighted_hops.append(len(path["path"]) - 1)
            weighted_distances.append(path["distance"])

# Geração de histogramas para o grafo ponderado
plt.hist(weighted_hops, bins=range(min(weighted_hops), max(weighted_hops) + 2), edgecolor='black', align='left')
plt.title("Hop Count (Weighted Graph)")
plt.xlabel("Number of Hops")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.hist(weighted_distances, bins=20, edgecolor='black')
plt.title("Distance in Kilometers (Weighted Graph)")
plt.xlabel("Distance (km)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Extração de hops e distâncias do grafo não ponderado
unweighted_hops = []
unweighted_distances = []
for path_set in unweighted_paths:
    for path in path_set:
        if path["distance"] < float("Inf"):  # Ignorar caminhos desconectados
            num_hops = len(path["path"]) - 1
            unweighted_hops.append(num_hops)
            unweighted_distances.append(num_hops)  # Ensure this is the number of hops


# Geração de histogramas para o grafo não ponderado
plt.hist(unweighted_hops, bins=range(min(unweighted_hops), max(unweighted_hops) + 2), edgecolor='black', align='left')
plt.xticks(range(min(unweighted_hops), max(unweighted_hops) + 1))  # Ensure integer ticks
plt.title("Hop Count (Unweighted Graph)")
plt.xlabel("Number of Hops")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.hist(unweighted_distances, bins=range(min(unweighted_distances), max(unweighted_distances) + 2), edgecolor='black', align='left')
plt.xticks(range(min(unweighted_distances), max(unweighted_distances) + 1))  # Ensure integer ticks
plt.title("Distance in Kilometers (Unweighted Graph)")
plt.xlabel("Distance (hops)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#########################################################################################################
#                             Terceira alinea da Segunda fase
#########################################################################################################

# Calcular a matriz de hops para o grafo ponderado
hop_matrix = countHops(unweighted_paths)

# Calcular o número médio de hops por demanda
total_hops = np.sum(hop_matrix)
n_nodes = len(matrix)

number_demands = n_nodes * (n_nodes - 1)
average_hops_per_demand = total_hops / number_demands
print("\n=== Auxiliar ===")
print(f"Total Hops: {total_hops}")
print(f"Number of Nodes: {n_nodes}")
print(f"Number of Demands N*(N-1): {number_demands}")


# Calcular o diâmetro da rede
network_diameter = np.max(hop_matrix)

# Estimação do número médio de hops
N = len(matrix)  # Número de nós
avg_hops_estimation = np.log(N) / np.log(average_degree)

# Imprimir resultados
print("\n=== Resultados da Terceira Alínea ===")
print(f"Número médio de hops por demanda: {average_hops_per_demand}")
print(f"Diâmetro da rede: {network_diameter}")
print(f"Estimativa do número médio de hops: {avg_hops_estimation}")

#########################################################################################################
#                                 Quarta alinea da Segunda fase
########################################################################################################## Node Connectivity
node_connectivity = calculate_node_connectivity(matrix)

# Edge Connectivity
edge_connectivity = calculate_edge_connectivity(matrix)

# Relação entre os parâmetros
print("\n=== Resultados da Quarta Alínea ===")
print(f"Relação entre parâmetros:")
print(f"Average Node Degree: {average_degree}")
print(f"Node Connectivity: {node_connectivity}")
print(f"Edge Connectivity: {edge_connectivity}")
print(f"Node Connectivity <= Average Node Degree: {node_connectivity <= average_degree}")
print(f"Edge Connectivity <= Node Connectivity: {edge_connectivity <= node_connectivity}")

#########################################################################################################
#                                 Quinta alinea da Segunda fase
#########################################################################################################

# Definir os nós x (Houston) e y (Seattle) pelo índice correspondente
x = 5  # Houston (índice 5 na matriz de adjacência)
y = 0  # Seattle (índice 0 na matriz de adjacência)

# Encontrar o corte mínimo de nós
node_cut_set = find_minimum_node_cut(matrix, x, y)

# Encontrar o corte mínimo de arestas
edge_cut_set = find_minimum_edge_cut(matrix, x, y)

print("\n=== Resultados da Quinta Alínea ===")
print(f"Minimum x-y Node Cut Set between Houston and Seattle: {node_cut_set}")
print(f"Minimum x-y Edge Cut Set between Houston and Seattle: {edge_cut_set}")

#########################################################################################################
#                                 Sexta alinea da Segunda fase
#########################################################################################################

# Nós x e y (Houston e Seattle)
x = 5  # Houston
y = 0  # Seattle

# Encontrar o caminho de serviço e os caminhos de backup
service_path, backup_paths = find_service_and_backup_paths(unweighted_paths, x, y)

# Exibir os resultados
print("\n=== Caminho de Serviço e Caminhos de Backup ===")
print(f"Caminho de Serviço entre Houston e Seattle: {service_path}")
print("Caminhos de Backup:")
for backup in backup_paths:
    print(backup) 
#########################################################################################################



#########################################################################################################
#                                 Third phase
#########################################################################################################


# Parameters for our network (Group 15)
X, Y, Z = 29, 46, 55

traffic_matrix = [
    [0, X, Y, Z, X, Y, 0, X, Y, 0, X, 0, 0, 0], # Node 1 -> Seattle
    [X, 0, X, Y, 0, X, 0, Z, X, Y, Z, 0, 0, Z], # Node 2 -> Palo Alto
    [Y, X, 0, X, Y, 0, X, 0, Z, X, 0, Z, 0, 0], # Node 3 -> San Diego
    [Z, Y, X, 0, X, Y, Z, 0, 0, Z, X, 0, Z, 0], # Node 4 -> Salt Lake City
    [X, 0, Y, X, 0, X, Y, Z, 0, Y, 0, 0, 0, Z], # Node 5 -> Boulder
    [Y, X, 0, Y, X, 0, X, Y, Z, X, Y, 0, X, 0], # Node 6 -> Houston
    [0, 0, X, Z, Y, X, 0, X, Y, Z, X, 0, Z, X], # Node 7 -> Lincoln
    [X, Z, 0, 0, Z, Y, X, 0, X, Y, Z, X, 0, Z], # Node 8 -> Champaign
    [Y, X, Z, 0, 0, Z, Y, X, 0, X, Y, Z, X, Y], # Node 9 -> Atlanta
    [0, Y, X, Z, Y, X, Z, Y, X, 0, X, Y, Z, X], # Node 10 -> Ann Arbor
    [X, Z, 0, X, 0, Y, X, Z, Y, X, 0, X, Y, Z], # Node 11 -> Pittsburgh
    [0, 0, Z, 0, 0, 0, 0, X, Z, Y, X, 0, X, Y], # Node 12 -> College Park
    [0, 0, 0, Z, 0, X, Z, 0, X, Z, Y, X, 0, X], # Node 13 -> Ithaca
    [0, Z, 0, 0, Z, 0, X, Z, Y, X, Z, Y, Y, 0], # Node 14 -> Princeton
]

#### Funcões auxiliares ####

def remove_link(matrix, i, j):
    """
    Remove a link between nodes i and j in the adjacency matrix.
    """
    matrix[i][j] = matrix[j][i] = 0
    return matrix

def calculate_demand_matrix(matrix):
    """
    Convert the traffic matrix into a demand matrix as a list of (source, destination, traffic) tuples.
    """
    demand_matrix = np.zeros_like(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > 0:  # Include only non-zero demands
                demand_matrix[i][j] = 1
    return demand_matrix

#### Primeira alinea ####

#remove link Colege Pk-Princeton from the matrix (physical topology)
x = 12 - 1  # College Park
y = 14 - 1  # Princeton

modified_matrix = remove_link(matrix_weighted, x, y)

demand_matrix = calculate_demand_matrix(traffic)

#Traffic
print("Traffic Matrix:")
print(traffic)

#Demand
print("\nDemand Matrix:")
print(demand_matrix)



#### Segunda alinea ####

# Visualize and save the link loads
def visualize_link_loads(link_loads, strategy_name, metric_name, output_dir="output"):
    """
    Create a bar chart to visualize link loads and save it to a file.
    
    Parameters:
    - link_loads: np.ndarray, the link load matrix
    - strategy_name: str, the name of the sorting strategy
    - metric_name: str, the name of the metric used
    - output_dir: str, directory where the chart will be saved
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(link_loads.flatten())), link_loads.flatten(), color='blue', alpha=0.7)
    plt.title(f"Link Loads ({metric_name.capitalize()} - {strategy_name.capitalize()})")
    plt.xlabel("Link Index")
    plt.ylabel("Load (Gb/s)")

    # Save the chart
    file_name = f"link_loads_{metric_name}_{strategy_name}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path, format="png", dpi=300)

    plt.show()
    print(f"Chart saved to {file_path}")

import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_link_loads_h(link_loads, strategy_name, metric_name, output_dir="output", top_n=None):
    """
    Create a horizontal bar chart to visualize non-zero link loads and save it to a file.
    
    Parameters:
    - link_loads: np.ndarray, the link load matrix
    - strategy_name: str, the name of the sorting strategy
    - metric_name: str, the name of the metric used
    - output_dir: str, directory where the chart will be saved
    - top_n: int, the number of top links to display (optional)
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Generate node IDs
    node_ids = [(i+1, j+1) for i in range(link_loads.shape[0]) for j in range(link_loads.shape[1])]
    
    # Filter links with load != 0
    non_zero_indices = np.where(link_loads.flatten() != 0)[0]
    non_zero_loads = link_loads.flatten()[non_zero_indices]
    non_zero_node_ids = [f"{node_ids[i][0]}-{node_ids[i][1]}" for i in non_zero_indices]
    
    # Sort by load (descending) if top_n is specified
    if top_n:
        sorted_indices = np.argsort(non_zero_loads)[::-1][:top_n]
        non_zero_loads = non_zero_loads[sorted_indices]
        non_zero_node_ids = [non_zero_node_ids[i] for i in sorted_indices]

    # Create horizontal bar chart
    plt.figure(figsize=(12, 12))  # Adjust figure height dynamically
    plt.bar(non_zero_node_ids, non_zero_loads, color='blue', alpha=0.7)
    plt.title(f"Link Loads ({metric_name.capitalize()} - {strategy_name.capitalize()})", fontsize=30)
    plt.xlabel("Load (Gb/s)", fontsize=16)
    plt.ylabel("Link (Node IDs)", fontsize=16)
    plt.xticks(fontsize=16, rotation=60)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=20)  # Reduce font size for labels

    # Set x-axis range
    plt.ylim(0, 750)

    # Save the chart
    file_name = f"link_loads_{metric_name}_{strategy_name}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.tight_layout()
    plt.savefig(file_path, format="png", dpi=300)

    plt.show()
    print(f"Chart saved to {file_path}")



# Compute shortest paths using a given metric
def compute_shortest_paths(graph, metric="hops"):
    """
    Compute shortest paths for all node pairs using Dijkstra's algorithm.
    """
    paths = {}
    for source in range(len(graph)):
        for target in range(len(graph)):
            if source != target:
                if metric == "hops":
                    # Find all shortest paths without considering edge weights
                    paths[(source, target)] = list(nx.all_shortest_paths(graph, source, target, weight=None))
                elif metric == "distance":
                    # Find all shortest paths considering edge weights
                    paths[(source, target)] = list(nx.all_shortest_paths(graph, source, target, weight="weight"))
    return paths

# Compute link loads using routing strategies with tie-breaking
def compute_link_loads_with_tie_breaking(graph, traffic_matrix, paths, sorting_strategy, link_loads):
    """
    Compute link loads for a given sorting strategy and traffic demands, considering tie-breaking.
    """
    demands = [(i, j, traffic_matrix[i][j]) for i in range(len(traffic_matrix)) for j in range(len(traffic_matrix[i])) if traffic_matrix[i][j] > 0]

    # Sort demands based on the strategy
    if sorting_strategy == "shortest-first":
        demands.sort(key=lambda x: len(paths[(x[0], x[1])]))  # Shortest paths first
        print("Shortest paths first")
        print(demands)
    elif sorting_strategy == "longest-first":
        demands.sort(key=lambda x: len(paths[(x[0], x[1])]), reverse=True)  # Longest paths first
        print("Longest paths first")
        print(demands)
    elif sorting_strategy == "largest-first":
        demands.sort(key=lambda x: x[2], reverse=True)  # Largest traffic first
        print("Largest traffic first")
        print(demands)

    # Route the demands considering tie-breaking
    for (source, target, traffic) in demands:
        all_paths = paths[(source, target)]  # All possible paths for the source-target pair
        best_path = None
        min_most_loaded_link = float("inf")

        # Tie-breaking: Choose the path that minimizes the load on the most loaded link
        for path in all_paths:
            trial_load = link_loads.copy()
            for u, v in zip(path[:-1], path[1:]):  # Simulate routing traffic
                trial_load[u][v] += traffic
            most_loaded_link = np.max(trial_load)
            if most_loaded_link < min_most_loaded_link:
                min_most_loaded_link = most_loaded_link
                best_path = path

        # Route traffic on the best path
        for u, v in zip(best_path[:-1], best_path[1:]):
            link_loads[u][v] += traffic

    return link_loads

# Main routine for solving the routing problem
def solve_uncapacitated_routing(physical_matrix, traffic_matrix, strategies, metrics):
    """
    Solve the uncapacitated routing problem for the provided strategies and metrics.
    """
    # Convert physical topology to graph
    physical_graph = nx.from_numpy_array(np.array(physical_matrix), create_using=nx.DiGraph)

    for metric in metrics:
        paths = compute_shortest_paths(physical_graph, metric=metric)
        for strategy in strategies:
            link_loads = np.zeros_like(physical_matrix)
            link_loads = compute_link_loads_with_tie_breaking(physical_graph, traffic_matrix, paths, strategy, link_loads)
            #visualize_link_loads(link_loads, strategy, metric)
            visualize_link_loads_h(link_loads, strategy, metric)

# Solve the problem
solve_uncapacitated_routing(
    physical_matrix=modified_matrix,
    traffic_matrix=traffic_matrix,
    strategies=["shortest-first", "longest-first", "largest-first"],
    metrics=["hops", "distance"]
)
        
exit()

#### Terceira alinea ####





#########################################################################################################

#CESNET
'''matrix = [[0,226.07,334.4,0,0,0,274.08],
          [226.07,0,315.98,0,0,0,0],
          [334.4,315.98,0,425.25,0,0,0],
          [0,0,425.25,0,378.51,173.75,0],
          [0,0,0,378.51,0,212.79,0],
          [0,0,0,173.75,212.79,0,330.72],
          [274.08,0,0,0,0,330.72,0]]'''
'''traffic = [[0,119.63,125.95,94.72,59.12,91.55,163.3],
          [119.63,0,95.83,83.72,54.12,79.97,114.80],
          [125.95,95.83,0.00,80.46,81.02,0.00,104.33],
          [94.72,83.72,80.46,0.00,116.99,144.81,114.30],
          [59.12,54.12,81.02,116.99,0.00,124.85,110.11],
          [91.55,79.97,0.00,144.81,124.85,0.00,144.93],
          [163.30,114.80,104.33,114.30,110.11,144.93,0.00]]
'''


#------------------------ INPUT SORTING ORDER----------------------------------------------
sorting_order = "shortest"
#sorting_order = "longest"
#sorting_order = "largest"
x
#-------------------------DETERMINATION OF NETWORK PARAMETERS-----------------------------

matrix_np = np.array(matrix,dtype=float)
#print(np.shape(matrix_np))
if not check_symmetric(matrix_np):
    print("ERROR: The input adjacency matrix (matrix) is not symmetric.")
    exit()
if not check_diagonal_zero(matrix_np):
    print("ERROR: The input adjacency matrix (matrix) has an element on the main diagonal different than zero.")
    exit()

start = time.time()

paths = shortestPaths(graph, matrix)

hop_matrix = countHops(paths)
average_node_degree.append(np.count_nonzero(matrix)/len(matrix))

N = matrix_np.shape[0]
number_of_links = np.count_nonzero(matrix_np)/2
min_link_length = np.min(matrix_np[np.nonzero(matrix_np)])
max_link_length = np.max(matrix_np)
matrix_np[matrix_np == 0] = np.nan
avg_link_length = np.nanmean(matrix_np)

traffic = create_traffic_matrix(matrix, traffic)
number_of_hops_per_demand.append(np.matrix(hop_matrix).sum() / np.matrix(traffic).sum())
diameter.append(np.matrix(hop_matrix).max())

ordered_paths = orderPaths(paths,traffic,hop_matrix, sorting_order)
load_matrix = create_load_matrix (matrix)
matrix_cp = matrix.copy()
load_matrix,path_matrix,distance_matrix,blocked_traffic, blocked_paths= route(ordered_paths, load_matrix, matrix_cp)

distance_matrix_np = np.array(distance_matrix,dtype=float)
distance_matrix_np[distance_matrix_np == 0] = np.nan
distance_matrix_np[distance_matrix_np >= 99999] = np.nan
min_path_length = np.nanmin(distance_matrix_np)
max_path_length = np.nanmax(distance_matrix_np)
avg_path_length = np.nanmean(distance_matrix_np)

total_num_paths = len(ordered_paths)
blocking_prob = len(blocked_paths)/total_num_paths

final_hop_matrix, avg_hops_per_path = hop_count(path_matrix)


# PRINTING RESULTS:
print("Number of Nodes:")
print(N)
print("Number of links:")
print(number_of_links)
print("Average Node Degree:")
print(average_node_degree)
print("Network Diameter:")
print(diameter)
print("Average Number of Hops per Demand:")
print(number_of_hops_per_demand)
print("Minimum link length:")
print(min_link_length)
print("Maximum link length:")
print(max_link_length)
print("Average link length:")
print(avg_link_length)
print("Total Number of Paths:")
print(len(ordered_paths))

average_load_per_link = printResults(ordered_paths,matrix,load_matrix,distance_matrix)

print("Average load per link:")
print(average_load_per_link)

print("Minimum path length:")
print(min_path_length)
print("Maximum path length:")
print(max_path_length)
print("Average path length:")
print(avg_path_length)
print("Average Number of Hops per Path (after routing):")
print(avg_hops_per_path)

print("Blocked Traffic:")
print(blocked_traffic)
print("Blocking Probability:")
print(blocking_prob)

for b in blocked_paths:
    print("'source':{},'destination':{},'path':{}".format(b['source'],b['destination'],b['path']))

end = time.time()
print("Runtime:")
print(end - start)