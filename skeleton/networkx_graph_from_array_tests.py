import nose.tools
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize

from skeleton.networkx_graph_from_array import get_networkx_graph_from_array


def _helper_networkx_graph(sample_image, expected_edges, expected_disjoint_graphs):
    networkx_graph = get_networkx_graph_from_array(sample_image)
    obtained_edges = networkx_graph.number_of_edges()
    obtained_disjoint_graphs = len(list(nx.connected_component_subgraphs(networkx_graph)))
    nose.tools.assert_greater_equal(obtained_edges, expected_edges)
    nose.tools.assert_equal(expected_disjoint_graphs, obtained_disjoint_graphs)


def test_tiny_loop_with_branches(size=(10, 10)):
    # a loop and a branches coming at end of the cycle
    frame = np.zeros(size, dtype=np.uint8)
    frame[2:-2, 2:-2] = 1
    frame[4:-4, 4:-4] = 0
    frame = skeletonize(frame)
    frame[1, 5] = 1
    frame[7, 5] = 1
    sample_image = np.zeros((3, 10, 10), dtype=np.uint8)
    sample_image[1] = frame
    _helper_networkx_graph(sample_image, 10, 1)


def test_disjoint_crosses(size=(10, 10, 10)):
    # two disjoint crosses
    cross_pair = np.zeros(size, dtype=np.uint8)
    cross = np.zeros((5, 5), dtype=np.uint8)
    cross[:, 2] = 1
    cross[2, :] = 1
    cross_pair[0, 0:5, 0:5] = cross
    cross_pair[5, 5:10, 5:10] = cross
    _helper_networkx_graph(cross_pair, 16, 2)


def test_single_voxel_line(size=(5, 5, 5)):
    sample_line = np.zeros(size, dtype=np.uint8)
    sample_line[1, :, 4] = 1
    _helper_networkx_graph(sample_line, 4, 1)


def test_special_case_graph():
    special_case_array = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                  [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                  [[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=bool)
    _helper_networkx_graph(special_case_array, 2, 1)
