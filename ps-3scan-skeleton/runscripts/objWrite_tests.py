import os
import shutil
import tempfile

from runscripts.objWrite import getObjBranchPointsWrite, getObjPointsWrite
from skeleton.skeleton_testlib import (get_cycles_with_branches_protrude, get_single_voxel_lineNobranches,
                                       get_cycle_no_tree, get_disjoint_trees_no_cycle_3d)


"""
Program to test if obj files written using objWrite.py) have expected lines starting with prefix 'l'
"""

FUNCTIONSTOTEST = [getObjPointsWrite, getObjBranchPointsWrite, ]


def _getLinePrefixes(graph, path):
    verticesList = []
    for function in FUNCTIONSTOTEST:
        # get number of lines in obj with prefix v
        tempDir = tempfile.mkdtemp() + os.sep
        tempDirPath = tempDir + path
        function(graph, tempDirPath)
        objFile = open(tempDirPath, "r")
        totalVertices = 0
        for line in objFile:
            for index, items in enumerate(line):
                if items == 'v':
                    totalVertices = totalVertices + 1
        shutil.rmtree(tempDir)
        verticesList.append(totalVertices)
    return verticesList


def test_singleSegment():
    # Test 1 Prefixes of v for a single segment
    lineGraph = get_single_voxel_lineNobranches()
    vertices = lineGraph.number_of_nodes()
    branchPoints = 0
    verticesList = _getLinePrefixes(lineGraph, "Line.obj")
    assert verticesList[0] == vertices, "number of vertices in single segment obj {}, not {}".format(verticesList[0], vertices)
    assert verticesList[1] == branchPoints, "number of branch vertices in single segment obj {}, not {}".format(verticesList[1], branchPoints)


def test_singleCycle():
    # Test 2 Prefixes of v for a single cycle
    donutGraph = get_cycle_no_tree()
    vertices = donutGraph.number_of_nodes()
    branchPoints = 0
    verticesListCycle = _getLinePrefixes(donutGraph, "OneCycle.obj")
    assert verticesListCycle[0] == vertices, "number of vertices in single cycle obj {}, not {}".format(verticesListCycle[0], vertices)
    assert verticesListCycle[1] == branchPoints, "number of branch vertices in single cycle obj {}, not {}".format(verticesListCycle[1], branchPoints)


def test_cycleAndTree():
    # Test 3 Prefixes of v for a cyclic tree
    sampleGraph = get_cycles_with_branches_protrude()
    vertices = sampleGraph.number_of_nodes()
    branchPoints = 2
    verticesListCyclicTree = _getLinePrefixes(sampleGraph, "CycleAndTree.obj")
    assert verticesListCyclicTree[0] == vertices, "number of vertices in cycleAndTree obj {}, not {}".format(verticesListCyclicTree[0], vertices)
    assert verticesListCyclicTree[1] == branchPoints, "number of branch vertices in cycleAndTree obj {}, not {}".format(verticesListCyclicTree[1], branchPoints)


def test_treeNoCycle3d():
    # Test 4 Prefixes of v for a tree
    crosPairgraph = get_disjoint_trees_no_cycle_3d()
    vertices = crosPairgraph.number_of_nodes()
    branchPoints = 2
    verticesListCrosses = _getLinePrefixes(crosPairgraph, "Tree.obj")
    assert verticesListCrosses[0] == vertices, "number of vertices in treeNoCycle3d obj {}, not {}".format(verticesListCrosses, vertices)
    assert verticesListCrosses[1] == branchPoints, "number of branch vertices in treeNoCycle3d obj {}, not {}".format(verticesListCrosses, branchPoints)

