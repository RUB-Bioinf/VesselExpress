from metrics.segmentStats import SegmentStats
from skeleton.skeleton_testlib import get_cycles_with_branches_protrude, get_single_voxel_lineNobranches, get_cycle_no_tree, get_disjoint_trees_no_cycle_3d

"""
Program to test if graphs created using networkxGraphFromArray and removeCliqueEdges
and from skeletons generated using the functions in the skeleton.skeletonClass, metrics.segmentStats
from the dictionary of the coordinate and adjacent nonzero coordinates
after removing the cliques have the number of segments as expected
PV TODO:Test if lengths of segments and tortuoisty of the curves as expected
"""


def test_cycleAndTree():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a cyclic tree
    sampleGraph = get_cycles_with_branches_protrude()
    stats = SegmentStats(sampleGraph)
    stats.setStats()
    assert stats.totalSegments == 4, "totalSegments in cycleAndTree sample should be 4, it is {}".format(stats.totalSegments)
    assert stats.typeGraphdict[0] == 3, "type of graph in cycleAndTree sample should be 3, it is {}".format(stats.typeGraphdict[0])
    assert stats.countEndPoints == 2, "number of end points in cycleAndTree sample should be 2, it is {}".format(stats.countEndPoints)
    assert stats.countBranchPoints == 2, "number of branch points in cycleAndTree sample should be 2, it is {}".format(stats.countBranchPoints)
    assert stats.cycleInfoDict[0][0] == 2, "number of branch points on the cycle must be 2, it is {}".format(stats.cycleInfoDict[0][0])


def test_singleSegment():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a single segment
    lineGraph = get_single_voxel_lineNobranches()
    stats = SegmentStats(lineGraph)
    stats.setStats()
    assert stats.totalSegments == 0, "totalSegments in singleSegment sample should be 0, it is {}".format(stats.totalSegments)
    assert stats.typeGraphdict[0] == 2, "type of graph in singleSegment sample should be 2, it is {}".format(stats.typeGraphdict[0])
    assert stats.countEndPoints == 2, "number of end points in singleSegment sample should be 2, it is {}".format(stats.countEndPoints)
    assert stats.countBranchPoints == 0, "number of branch points in singleSegment sample should be 0, it is {}".format(stats.countBranchPoints)
    assert stats.hausdorffDimensionDict == {}, "hausdorffDimensionDict must be empty, it is {}".format(stats.hausdorffDimensionDict)
    assert stats.cycleInfoDict == {}, "cycleInfoDict must be empty, it is {}".format(stats.cycleInfoDict)


def test_singleCycle():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a single cycle
    donutGraph = get_cycle_no_tree()
    stats = SegmentStats(donutGraph)
    stats.setStats()
    assert stats.totalSegments == 1, "totalSegments in singleCycle sample should be 1, it is {}".format(stats.totalSegments)
    assert stats.typeGraphdict[0] == 1, "type of graph in singleCycle sample should be 1, it is {}".format(stats.typeGraphdict[0])
    assert stats.countEndPoints == 0, "number of end points in singleCycle sample should be 2, it is {}".format(stats.countEndPoints)
    assert stats.countBranchPoints == 0, "number of branch points in singleCycle sample should be 0, it is {}".format(stats.countBranchPoints)
    assert stats.hausdorffDimensionDict == {}, "hausdorffDimensionDict must be empty, it is {}".format(stats.hausdorffDimensionDict)
    assert stats.cycleInfoDict[0][0] == 0, "number of branch points on the cycle must be 0, it is {}".format(stats.cycleInfoDict[0][0])


def test_treeNoCycle3D():
    # test if stats i.e segments, type of graph. branch, end points, and information about cycle
    # is as expected for a tree like structure
    crosPairgraph = get_disjoint_trees_no_cycle_3d()
    stats = SegmentStats(crosPairgraph)
    stats.setStats()
    assert stats.totalSegments == 8, "totalSegments in treeNoCycle3D sample should be 8, it is {}".format(stats.totalSegments)
    assert stats.typeGraphdict[0] == 4, "type of graph in treeNoCycle3D sample should be 4, it is {}".format(stats.typeGraphdict[0])
    assert stats.countEndPoints == 8, "number of end points in treeNoCycle3D sample should be 2, it is {}".format(stats.countEndPoints)
    assert stats.countBranchPoints == 2, "number of branch points in treeNoCycle3D sample should be 2, it is {}".format(stats.countBranchPoints)
    assert stats.cycleInfoDict == {}, "cycleInfoDict must be empty, it is {}".format(stats.cycleInfoDict)


