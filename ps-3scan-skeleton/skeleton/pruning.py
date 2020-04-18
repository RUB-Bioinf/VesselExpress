import itertools
import time

import networkx as nx
"""
program to prune segments of length less than cutoff in  a 3D/2D Array
"""


def _countBranchPointsOnSimplePath(simplePath, listBranchIndices):
    """
    Find number of branch points on a path
    Parameters
    ----------
    simplePath : list
        list of nodes on the path

    listBranchIndices : list
        list of branch nodes

    Returns
    -------
    integer
        number of branch nodes on the path
    """
    return sum([1 for point in simplePath if point in listBranchIndices])


def _removeNodesOnPath(simplePaths, skel):
    """
    Returns array changed in place after zeroing out the nodes on simplePath
    Parameters
    ----------
    simplePath : list
        list of list of nodes on simple paths

    skel : numpy array
        2D or 3D numpy array

    """
    for simplePath in simplePaths:
        for pointsSmallBranches in simplePath[1:]:
            skel[pointsSmallBranches] = 0


def getPrunedSkeleton(skeletonStack, networkxGraph, cutoff=9):
    """
    Returns an array changed in place with segments less than cutoff removed
    Parameters
    ----------
    skeletonStack : numpy array
        2D or 3D numpy array

    networkxGraph : Networkx graph
        graph to remove cliques from

    cutoff : integer
        cutoff of segment length to be removed

    """
    start_prune = time.time()
    ndd = nx.degree(networkxGraph)
    listEndIndices = [k for (k, v) in ndd.items() if v == 1]
    listBranchIndices = [k for (k, v) in ndd.items() if v != 2 and v != 1]
    branchEndPermutations = list(itertools.product(listEndIndices, listBranchIndices))
    totalSteps = len(branchEndPermutations)
    for index, (endPoint, branchPoint) in enumerate(branchEndPermutations):
        if nx.has_path(networkxGraph, endPoint, branchPoint):  # is it on the same subgraph
            simplePaths = [simplePath for simplePath in nx.all_simple_paths(networkxGraph, source=endPoint,
                           target=branchPoint, cutoff=cutoff) if _countBranchPointsOnSimplePath(simplePath, listBranchIndices)]
            _removeNodesOnPath(simplePaths, skeletonStack)
        progress = int((100 * (index + 1)) / totalSteps)
        print("pruning in progress {}% \r".format(progress), end="", flush=True)
    print("time taken to prune is %0.3f seconds" % (time.time() - start_prune))
    return skeletonStack
