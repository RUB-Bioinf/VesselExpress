import numpy as np
import networkx as nx

"""
write networkxgraph of skeleton to a wavefront obj file - primarily written
for netmets (comparison of 2 networks)
Link to the software - http://stim.ee.uh.edu/resources/software/netmets/
Currently used as an input for visualization team
"""


def _removeEdgesInVisitedPath(subGraphSkeleton, path, isCycle):
    """
    Remove edges belonging to "path" from "subGraphSkeleton"
    Parameters
    ----------
    subGraphSkeleton : Networkx graph
        input graph to remove edges from

    path : list
       list of nodes in the path

    isCycle : boolean
       Specify if path is a cycle or not

    Returns
    -------
    subGraphSkeleton : Networkx graph
        graph changed inplace with visited edges removed

    Notes
    ------
    given a visited path in variable path, the edges in the
    path are removed in the graph "subGraphSkeleton".
    if isCycle = 1 , the given path belongs to a cycle,
    so an additional edge is formed between the last and the
    first node to form a closed cycle/ path and is removed
    """
    shortestPathEdges = []
    for index, item in enumerate(path):
        if index + 1 != len(path):
            item2 = path[index + 1]
        elif isCycle:
            item2 = path[0]
        shortestPathEdges.append(tuple((item, item2)))
    subGraphSkeleton.remove_edges_from(shortestPathEdges)


def getObjBranchPointsWrite(networkxGraph, pathTosave):
    """
    Writes a networkx graph's branch points to an obj file
    Parameters
    ----------
    networkxGraph : Networkx graph
        graph to be converted to obj

    pathTosave : str
        write the obj file at pathTosave

    Returns
    -------
    Writes a networkx graph's branch points to an obj file at pathTosave

    Notes
    -----
    Expects aspect ratio of array to be pre-adjusted
    """
    objFile = open(pathTosave, "w")  # open a obj file in the given path
    nodeDegreedict = nx.degree(networkxGraph)
    branchpoints = [k for (k, v) in nodeDegreedict.items() if v != 2 and v != 1]
    #  for each of the sorted vertices
    strsVertices = []
    for index, vertex in enumerate(branchpoints):
        strsVertices.append("v " + " ".join(str(vertex[i]) for i in [1, 0, 2]) + "\n")  # add strings of vertices to obj file
    objFile.writelines(strsVertices)  # write strings to obj file
    objFile.close()


def getObjPointsWrite(networkxGraph, pathTosave):
    """
    Writes a networkx graph nodes of a skeleton as vertices to an obj file
    Parameters
    ----------
    networkxGraph : Networkx graph
        graph to be converted to obj

    pathTosave : str
        write the obj file at pathTosave

    Returns
    -------
    Writes a networkx graph nodes of a skeleton as vertices to an obj file at pathTosave
    Notes
    -----
    Expects aspect ratio of array to be pre-adjusted
    """
    objFile = open(pathTosave, "w")  # open a obj file in the given path
    nodes = nx.nodes(networkxGraph)
    strsVertices = []
    for index, vertex in enumerate(nodes):
        strsVertices.append("v " + " ".join(str(vertex[i]) for i in [1, 0, 2]) + "\n")  # add strings of vertices to obj file
    objFile.writelines(strsVertices)  # write strings to obj file
    objFile.close()


if __name__ == '__main__':
    # read points into array
    skeletonIm = np.load(input("enter a path to shortest path skeleton volume------"))
    aspectRatio = input("please enter resolution of a voxel in 3D with resolution in z followed by y and x")
    aspectRatio = [float(item) for item in aspectRatio.split(' ')]
    path = input("please enter a path to save resultant obj file with no texture coordinates")
    getObjPointsWrite(skeletonIm, path, aspectRatio)
