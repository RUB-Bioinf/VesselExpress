import itertools
import time
from enum import Enum

import networkx as nx
import numpy as np


"""
Find the segments, lengths and tortuosity of a networkx graph by
        1) go through each of the disjoint graphs
        2) decide if it is one of the following a) line
        b) cycle c) acyclic tree like structure d) cyclic tree like structure
        e) single node
        3) Find all the paths in a given disjoint graph from a point whose degree is greater
        than 2 (branch point) to a point whose degree equal to one (end point)
        4) calculate distance between edges in each path and displacement to find curve length and
        curve displacement to find tortuosity, hausdorff dimension and contraction
        5) Remove all the edges in this path once they are traced
"""


class SubgraphTypes(Enum):
    """
    Enumerate typeGraphdict's type of subgraphs and values
    """
    singleNode = 0
    singleCycle = 1
    singleLine = 2
    cyclic = 3
    acyclic = 4


class SegmentStats:
    """
    Find statistics on a networkx graph of a skeleton
    Parameters
    ----------
    Graph : networkx graph
       networkx graph of a skeleton

    Examples
    ------
           SegmentStats.countDict - A dictionary with key as the node(branch or end point)
                                          and number of branches connected to it

           SegmentStats.lengthDict - A dictionary with key as the (branching index (nth branch from the start node),
                                           start node, end node of the branch) value = length of the branch

           SegmentStats.tortuosityDict - A dictionary with key as the (branching index (nth branch from the start node),
                                             start node, end node of the branch) value = tortuosity of the branch

           SegmentStats.totalSegments - total number of branches (segments between branch and branch points, branch and
                                        end points)

           SegmentStats.typeGraphdict - A dictionary with the nth disjoint graph as the key and the type of graph as
                                        the value

           SegmentStats.avgBranching - Average branching index (number of branches at a branch point) of the network

           SegmentStats.countEndPoints- nodes with only one other node connected to them

           SegmentStats.countBranchPoints - nodes with more than 2 nodes connected to them

           SegmentStats.contractionDict - A dictionary with key as the (branching index (nth branch from the start node)
                                          ,start node, end node of the branch) value = contraction of the branch

           SegmentStats.hausdorffDimensionDict - A dictionary with key as the (branching index (nth branch from
                                                 the start node), start node,
                                                 end node of the branch) value = hausdorff dimension of the branch

           SegmentStats.cycleInfoDict - Dictionary with key as nth cycle, value as list with [number of branch points on
                                        cycle, length of the cycle]

           SegmentStats.isolatedEdgeInfoDict - A dictionary with key as the (start node, end node of the branch), value
                                               as the length of the segment
           isolatededges are the edges that form a single segment of line with no trees in them, except if it's
           a dichtonomous tree with segments of
           length 1 at 45 degrees


    Notes
    --------

    tortuosity = curveLength / curveDisplacement
    contraction = curveDisplacement / curveLength (better becuase there is no change of instability (undefined) in case
                                                    of cycles)
    Hausdorff Dimension = np.log(curveLength) / np.log(curveDisplacement)
    https://en.wikipedia.org/wiki/Hausdorff_dimension
    Type of subgraphs:
    0 = if graph a single node
    1 = if graph is a single cycle
    2 = line (highest degree in the subgraph is 2)
    3 = undirected acyclic graph
    4 = undirected cyclic graph

    """
    def __init__(self, networkxGraph):
        self.networkxGraph = networkxGraph
        # intitialize all the instance variables of SegmentStats class
        self.contractionDict = {}
        self.countDict = {}
        self.cycleInfoDict = {}
        self.hausdorffDimensionDict = {}
        self.isolatedEdgeInfoDict = {}
        self.lengthDict = {}
        self.tortuosityDict = {}
        self.typeGraphdict = {}

        self.cycles = 0
        self.edgesUntraced = 0
        self.totalSegments = 0
        self._sortedSegments = []
        self._visitedSources = []
        self._visitedPaths = []
        # list of _disjointGraphs
        self._disjointGraphs = list(nx.connected_component_subgraphs(self.networkxGraph))

    def _getLengthAndRemoveTracedPath(self, path, isCycle=False, remove=True):
        """
        Find length of a path as distance between nodes in it
        and Remove edges belonging to "path" from "self._subGraphSkeleton" by default
        Parameters
        ----------
        self._subGraphSkeleton : Networkx graph

        path : list
           list of nodes in the path

        isCycle : boolean
           Specify if path is a cycle or not, default not a cycle

        remove : boolean
            Specifies whether or not to remove a path from self._subGraphSkeleton

        Returns
        -------
        self._subGraphSkeleton : Networkx graph
            graph changed inplace with visited edges removed

        length : float
            Length of path

        Notes
        ------
        "Given a visited path in variable path, the edges in the
        path are removed in the graph
        if isCycle = 1 , the given path belongs to a cycle,
        so an additional edge is formed between the last and the
        first node to form a closed cycle/ path and the edges in the path
        are removed
        If remove is False, returns the length but the path is not removed,
        else removes the path and returns the length as well"

        Examples
        --------
        >>> path = [(1, 2, 3), (1, 3, 3), (1, 4, 5)]
        >>> length = self._getLengthAndRemoveTracedPath(path)
        >>> length
        3.2360679774997898
        >>> lengthCycle = self._getLengthAndRemoveTracedPath(path, 1)
        6.0644951022459797
        """
        length = 0
        shortestPathEdges = []
        for index, item in enumerate(path):
            if index + 1 != len(path):
                item2 = path[index + 1]
            elif isCycle:
                item2 = path[0]
            vect = [j - i for i, j in zip(item, item2)]
            length += np.linalg.norm(vect)
            shortestPathEdges.append(tuple((item, item2)))
        if remove:
            self._subGraphSkeleton.remove_edges_from(shortestPathEdges)
        return length

    def _checkSegmentNotTraced(self, simplePath):
        """
        Find if simplePath is not already been visited for sorted segments of a cycle

        Parameters
        ----------
        simplePath: list
           list of nodes in the path

        Returns
        -------
        True if simplePath is in none of sortedSegments (intersection <=2)
        False if any one of the simplePath is in sortedSegments

        Notes
        -----
        _sortedSegments: list of lists
           list of paths already traced
        """
        return all(len(set(p) & set(simplePath)) <= 2 for p in self._sortedSegments)

    def _singleSegment(self, nodes):
        # disjoint line or a bent line at 45 degrees appearing as dichtonomous tree but an error due to
        # improper binarization, so remove them and do not account for statistics
        listOfPerms = list(itertools.combinations(nodes, 2))
        if type(nodes[0]) == int:
            modulus = [[start - end] for start, end in listOfPerms]
            dists = [abs(i[0]) for i in modulus]
        else:
            dims = len(nodes[0])
            modulus = [[start[dim] - end[dim] for dim in range(0, dims)] for start, end in listOfPerms]
            dists = [sum(modulus[i][dim] * modulus[i][dim] for dim in range(0, dims)) for i in range(0, len(modulus))]
        if len(list(nx.articulation_points(self._subGraphSkeleton))) == 1 and set(dists) != 1:
            # each node is connected to one or two other nodes which are not a distance of 1 implies there is a
            # one branch point with two end points in a single dichotomous tree"""
            for sourceOnTree, item in listOfPerms:
                if nx.has_path(self._subGraphSkeleton, sourceOnTree, item) and sourceOnTree != item:
                    simplePaths = list(nx.all_simple_paths(self._subGraphSkeleton, source=sourceOnTree, target=item))
                    simplePath = simplePaths[0]
                    countBranchNodesOnPath = sum([1 for point in simplePath if point in nodes])
                    if countBranchNodesOnPath == 2:
                        curveLength = self._getLengthAndRemoveTracedPath(simplePath)
                        self.isolatedEdgeInfoDict[sourceOnTree, item] = curveLength
        else:
            # each node is connected to one or two other nodes implies it is a line,
            endPoints = [k for (k, v) in self._nodeDegreeDict.items() if v == 1]
            sourceOnLine = endPoints[0]
            targetOnLine = endPoints[1]
            simplePath = nx.shortest_path(self._subGraphSkeleton, source=sourceOnLine, target=targetOnLine)
            curveLength = self._getLengthAndRemoveTracedPath(simplePath)
            self.isolatedEdgeInfoDict[sourceOnLine, targetOnLine] = curveLength

    def _setCountDict(self, source):
        if source not in self._visitedSources:
            self.countDict[source] = 1
            self._visitedSources.append(source)
        else:
            self.countDict[source] = self.countDict[source] + 1

    def _setHausdorffDimensionDict(self, source, target, length, displacement):
        logDisplacement = np.log(displacement)
        if logDisplacement != -np.inf or not np.allclose(logDisplacement, 0.0):
            if not logDisplacement == 0.0:
                self.hausdorffDimensionDict[self.countDict[source], source, target] = np.log(length) / logDisplacement

    def _singleCycle(self, cycle):
        """
        Find statistics of a single cycle of a disjoint graph
        """
        sourceOnCycle = cycle[0]
        self._setCountDict(sourceOnCycle)
        curveLength = self._getLengthAndRemoveTracedPath(cycle, isCycle=True)
        self.cycleInfoDict[self.cycles] = [0, curveLength]
        nthCycle = self.countDict[sourceOnCycle]
        self.lengthDict[nthCycle, sourceOnCycle, cycle[len(cycle) - 1]] = curveLength
        self.tortuosityDict[nthCycle, sourceOnCycle, cycle[len(cycle) - 1]] = 0
        self.contractionDict[nthCycle, sourceOnCycle, cycle[len(cycle) - 1]] = 0
        self.cycles = self.cycles + 1

    def _cyclicTree(self, cycleList):
        """
        Find statistics of a cyclic tree of a disjoint graph
        """
        for nthcycle, cycle in enumerate(cycleList):
            nodeDegreeDictFilt = {key: value for key, value in self._nodeDegreeDict.items() if key in cycle}
            branchPointsOnCycle = [k for (k, v) in nodeDegreeDictFilt.items() if v != 2 and v != 1]
            sourceOnCycle = branchPointsOnCycle[0]
            if len(branchPointsOnCycle) == 1:
                self._singleCycle(cycle)
            else:
                for point in cycle:
                    if point not in branchPointsOnCycle:
                        continue
                    if not (nx.has_path(self._subGraphSkeleton, source=sourceOnCycle, target=point) and
                            sourceOnCycle != point):
                        continue
                    simplePath = nx.shortest_path(self._subGraphSkeleton, source=sourceOnCycle, target=point)
                    sortedSegment = sorted(simplePath)
                    isSegmentTraced = (sortedSegment not in self._sortedSegments and
                                       self._checkSegmentNotTraced(simplePath))
                    countBranchNodesOnPath = sum([1 for pathPoint in simplePath if pathPoint in branchPointsOnCycle])
                    if countBranchNodesOnPath == 2 and isSegmentTraced:
                        self._setCountDict(sourceOnCycle)
                        curveLength = self._getLengthAndRemoveTracedPath(simplePath, remove=False)
                        curveDisplacement = np.sqrt(np.sum((np.array(sourceOnCycle) - np.array(point)) ** 2))
                        nthSegment = self.countDict[sourceOnCycle]
                        self.lengthDict[nthSegment, sourceOnCycle, point] = curveLength
                        self.tortuosityDict[nthSegment, sourceOnCycle, point] = curveLength / curveDisplacement
                        self.contractionDict[nthSegment, sourceOnCycle, point] = curveDisplacement / curveLength
                        self._setHausdorffDimensionDict(sourceOnCycle, point, curveLength, curveDisplacement)
                        self._visitedPaths.append(simplePath)
                        self._sortedSegments.append(sortedSegment)
                    sourceOnCycle = point
            self.cycleInfoDict[self.cycles] = [len(branchPointsOnCycle),
                                               self._getLengthAndRemoveTracedPath(cycle, isCycle=True, remove=False)]
            self.cycles += 1
        for path in self._visitedPaths:
            self._getLengthAndRemoveTracedPath(path)
        self._tree()

    def _getStatsTree(self, listOfPerms, intersection):
        for sourceOnTree, item in listOfPerms:
            if not (nx.has_path(self._subGraphSkeleton, sourceOnTree, item) and sourceOnTree != item):
                continue
            simplePaths = list(nx.all_simple_paths(self._subGraphSkeleton, source=sourceOnTree, target=item))
            for simplePath in simplePaths:
                countBranchNodesOnPath = sum([1 for point in simplePath if point in self._branchPoints])
                if not(countBranchNodesOnPath == intersection):
                    continue
                self._setCountDict(sourceOnTree)
                curveLength = self._getLengthAndRemoveTracedPath(simplePath)
                curveDisplacement = np.sqrt(np.sum((np.array(sourceOnTree) - np.array(item)) ** 2))
                nthSegment = self.countDict[sourceOnTree]
                self.lengthDict[nthSegment, sourceOnTree, item] = curveLength
                self.tortuosityDict[nthSegment, sourceOnTree, item] = curveLength / curveDisplacement
                self.contractionDict[nthSegment, sourceOnTree, item] = curveDisplacement / curveLength
                self._setHausdorffDimensionDict(sourceOnTree, item, curveLength, curveDisplacement)

    def _tree(self):
        """
        Find statistics of a tree like structure of disjoint graph
        """
        listOfPerms = list(itertools.product(self._branchPoints, self._endPoints))
        self._getStatsTree(listOfPerms, 1)

    def _branchToBranch(self):
        """
        Find untraced edges between two branch points
        """
        listOfPerms = list(itertools.permutations(self._branchPoints, 2))
        self._getStatsTree(listOfPerms, 2)

    def _findAccessComponentsDisjoint(self):
        self._nodes = self._subGraphSkeleton.nodes()
        self._nodeDegreeDict = nx.degree(self._subGraphSkeleton)
        self._branchPoints = [k for (k, v) in self._nodeDegreeDict.items() if v != 2 and v != 1]
        self._endPoints = [k for (k, v) in self._nodeDegreeDict.items() if v == 1]
        self._branchPoints.sort()
        self._endPoints.sort()
        self._cycleList = nx.cycle_basis(self._subGraphSkeleton)
        self._cycleCount = len(self._cycleList)
        self._degreeList = list(self._nodeDegreeDict.values())

    def _findAccessComponentsNetworkx(self):
        self.totalSegments = len(self.lengthDict)
        listCounts = list(self.countDict.values())
        self.avgBranching = 0
        if len(self.countDict) != 0:
            self.avgBranching = sum(listCounts) / len(self.countDict)
        ndd = nx.degree(self.networkxGraph)
        self.countEndPoints = sum([1 for key, value in ndd.items() if value == 1])
        self.countBranchPoints = sum([1 for key, value in ndd.items() if value > 2])

    def setStats(self):
        """1) go through each of the disjoint graphs
           2) decide if it is one of the following a) single node b) cycle
           c) line d) cyclic tree like structure e) acyclic tree like structure
           3) And set stats for each subgraph
           if edges are still untraced, then enumerate and remove them paths using
            self._branchToBranch
        """
        start = time.time()
        countDisjointGraphs = len(self._disjointGraphs)
        for self._ithDisjointGraph, self._subGraphSkeleton in enumerate(self._disjointGraphs):
            self._findAccessComponentsDisjoint()
            if len(self._nodes) == 1:
                self.typeGraphdict[self._ithDisjointGraph] = SubgraphTypes.singleNode.value
            elif (min(self._degreeList) == max(self._degreeList) and
                  nx.is_biconnected(self._subGraphSkeleton) and
                  self._cycleCount == 1):
                self._singleCycle(self._cycleList[0])
                self.typeGraphdict[self._ithDisjointGraph] = SubgraphTypes.singleCycle.value
            elif set(self._degreeList) == set((1, 2)) or set(self._degreeList) == {1}:
                self._singleSegment(self._nodes)
                self.typeGraphdict[self._ithDisjointGraph] = SubgraphTypes.singleLine.value
            elif self._cycleCount != 0:
                self._cyclicTree(self._cycleList)
                self.typeGraphdict[self._ithDisjointGraph] = SubgraphTypes.cyclic.value
            else:
                self._tree()
                self.typeGraphdict[self._ithDisjointGraph] = SubgraphTypes.acyclic.value
            # check if any unfinished business in _subGraphSkeleton, untraced edges
            if self._subGraphSkeleton.number_of_edges() != 0:
                self._branchToBranch()
            assert self._subGraphSkeleton.number_of_edges() == 0, ("edges not removed are"
                                                                   "%i" % self._subGraphSkeleton.number_of_edges())
            progress = int((100 * self._ithDisjointGraph) / countDisjointGraphs)
            print("finding segment stats in progress {}% \r".format(progress), end="", flush=True)
            if True:
                print()
        self._findAccessComponentsNetworkx()
        print("time taken to calculate segments and their lengths is %0.3f seconds" % (time.time() - start))
