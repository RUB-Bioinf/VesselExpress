import itertools
import time

import numpy as np
import networkx as nx
from scipy.ndimage import convolve


"""
program to look up adjacent elements and calculate degree
this dictionary can be used for graph creation
since networkx graph based on looking up the array and the
adjacent coordinates takes long time. create a dict
using dict_of_indices_and_adjacent_coordinates.
(-1 -1 -1) (-1 0 -1) (-1 1 -1)
(-1 -1 0)  (-1 0 0)  (-1 1 0)
(-1 -1 1)  (-1 0 1)  (-1 1 1)

(0 -1 -1) (0 0 -1) (0 1 -1)
(0 -1 0)  (0 0 0)  (0 1 0)
(0 -1 1)  (0 0 1)  (0 1 1)

(1 -1 -1) (1 0 -1) (1 1 -1)
(1 -1 0)  (1 0 0)  (1 1 0)
(1 -1 1)  (1 0 1)  (1 1 1)
"""

# permutations of (-1, 0, 1) in three/two dimensional tuple format
# representing 8 and 26 increments around a pixel at origin (0, 0, 0)
# 2nd ordered neighborhood around a voxel/pixel
LIST_STEP_DIRECTIONS3D = list(itertools.product((-1, 0, 1), repeat=3))
LIST_STEP_DIRECTIONS3D.remove((0, 0, 0))

LIST_STEP_DIRECTIONS2D = list(itertools.product((-1, 0, 1), repeat=2))
LIST_STEP_DIRECTIONS2D.remove((0, 0))


def _get_increments(config_number, dimensions):
    """
    Return position of non zero voxels/pixels in the
    binary string of config number
    Parameters
    ----------
    config_number : int64
        integer less than 2 ** 26

    dimensions: int
        number of dimensions, can only be 2 or 3

    Returns
    -------
    list
        a list of incremental direction of a non zero voxel/pixel

    Notes
    ------
    As in the beginning of the program, there are incremental directions
    around a voxel at origin (0, 0, 0) which are returned by this function.
    config_number is a decimal number representation of 26 binary numbers
    around a voxel at the origin in a second ordered neighborhood
    """
    config_number = np.int64(config_number)
    if dimensions == 3:
        # convert decimal number to a binary string
        list_step_directions = LIST_STEP_DIRECTIONS3D
    elif dimensions == 2:
        list_step_directions = LIST_STEP_DIRECTIONS2D
    neighbor_values = [(config_number >> digit) & 0x01 for digit in range(3 ** dimensions - 1)]
    return [neighbor_value * increment for neighbor_value, increment in zip(neighbor_values, list_step_directions)]


def _set_adjacency_list(arr):
    """
    Return position of non zero voxels/pixels in the
    binary string of config number
    Parameters
    ----------
    arr : numpy array
        binary numpy array can only be 2D Or 3D

    Returns
    -------
    dict_of_indices_and_adjacent_coordinates: Dictionary
        key is the nonzero coordinate in input "arr" and value
        is all the position of nonzero coordinates around it
        in it's second order neighborhood

    """
    dimensions = arr.ndim
    assert dimensions in [2, 3], "array dimensions must be 2 or 3, they are {}".format(dimensions)
    if dimensions == 3:
        # flipped 3D template in advance
        template = np.array([[[33554432, 16777216, 8388608], [4194304, 2097152, 1048576], [524288, 262144, 131072]],
                            [[65536, 32768, 16384], [8192, 0, 4096], [2048, 1024, 512]],
                            [[256, 128, 64], [32, 16, 8], [4, 2, 1]]], dtype=np.uint64)
    else:
        # 2 dimensions
        template = np.array([[128, 64, 32], [16, 0, 8], [4, 2, 1]], dtype=np.uint64)
    # convert the binary array to a configuration number array of same size
    # by convolving with template
    arr = np.ascontiguousarray(arr, dtype=np.uint64)
    result = convolve(arr, template, mode='constant', cval=0)
    # set the values in convolution result to zero which were zero in 'arr'
    result[arr == 0] = 0
    dict_of_indices_and_adjacent_coordinates = {}
    # list of nonzero tuples
    non_zeros = list(set(map(tuple, np.transpose(np.nonzero(arr)))))
    if np.sum(arr) == 1:
        # if there is just one nonzero element there are no adjacent coordinates
        dict_of_indices_and_adjacent_coordinates[non_zeros[0]] = []
    else:
        for item in non_zeros:
            adjacent_coordinate_list = [tuple(np.array(item) + np.array(increments))
                                        for increments in _get_increments(result[item], dimensions) if increments != ()]
            dict_of_indices_and_adjacent_coordinates[item] = adjacent_coordinate_list
    return dict_of_indices_and_adjacent_coordinates


def _remove_clique_edges(networkx_graph):
    """
    Return 3 vertex clique removed graph
    Parameters
    ----------
    networkx_graph : Networkx graph
        graph to remove cliques from

    Returns
    -------
    networkx_graph: Networkx graph
        graph with 3 vertex clique edges removed

    Notes
    ------
    Removes the longest edge in a 3 Vertex cliques,
    Special case edges are the edges with equal
    lengths that form the 3 vertex clique.
    Doesn't deal with any other cliques
    """
    start = time.time()
    cliques = nx.find_cliques_recursive(networkx_graph)
    # all the nodes/vertices of 3 cliques
    three_vertex_cliques = [clq for clq in cliques if len(clq) == 3]
    if len(list(three_vertex_cliques)) != 0:
        combination_edges = [list(itertools.combinations(clique, 2)) for clique in three_vertex_cliques]
        subgraph_edge_lengths = []
        # different combination of edges in the cliques and their lengths
        for combinationEdge in combination_edges:
            subgraph_edge_lengths.append([np.sum((np.array(item[0]) - np.array(item[1])) ** 2)
                                          for item in combinationEdge])
        clique_edges = []
        # clique edges to be removed are collected here
        # the edges with maximum edge length
        for main_dim, item in enumerate(subgraph_edge_lengths):
            if len(set(item)) != 1:
                for sub_dim, length in enumerate(item):
                    if length == max(item):
                        clique_edges.append(combination_edges[main_dim][sub_dim])
            else:
                special_case = combination_edges[main_dim]
                diff_of_edges = []
                for num_spcl_edges in range(0, 3):
                    source = list(special_case[num_spcl_edges][0])
                    target = list(special_case[num_spcl_edges][1])
                    diff_of_edges.append([i - j for i, j in zip(source, target)])
                for index, val in enumerate(diff_of_edges):
                    if val[0] == 0:
                        sub_dim = index
                        clique_edges.append(combination_edges[main_dim][sub_dim])
                        break
        networkx_graph.remove_edges_from(clique_edges)
        print("time taken to remove cliques is %0.2f seconds" % (time.time() - start))
    return networkx_graph


def get_networkx_graph_from_array(binary_arr):
    """
    Return a networkx graph from a binary numpy array
    Parameters
    ----------
    binary_arr : numpy array
        binary numpy array can only be 2D Or 3D

    Returns
    -------
    networkx_graph : Networkx graph
        graphical representation of the input array after clique removal
    """
    assert np.max(binary_arr) in [0, 1], "input must always be a binary array"
    start = time.time()
    dict_of_indices_and_adjacent_coordinates = _set_adjacency_list(binary_arr)
    networkx_graph = nx.from_dict_of_lists(dict_of_indices_and_adjacent_coordinates)
    _remove_clique_edges(networkx_graph)
    print("time taken to obtain networkxgraph is %0.3f seconds" % (time.time() - start))
    return networkx_graph
