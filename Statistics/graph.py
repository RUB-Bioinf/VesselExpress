import networkx as nx
from collections import defaultdict
import time
import Statistics.filament as fil
from scipy import ndimage as ndi
import Statistics.measurements as ms

class Graph:
    """
        Find statistics on a graph of a skeleton

        Parameters
        ----------
        graph : networkx graph
            networkx graph of a skeleton

        Examples
        --------
        Graph.segmentsTotal - total number of segments (branches between branch/end point and branch/end point)

        Graph.segmentsDict - A dictionary with the nth disjoint graph as the key containing a dictionary
                                with key as the segment index (start node, end node) and value = list of nodes

        Graph.lengthDict - A dictionary with the nth disjoint graph as the key containing a dictionary
                                with key as the segment index (start node, end node) and value = length of the segment

        Graph.sumLengthDict - A dictionary with the nth disjoint graph as the key and the total length as the value

        Graph.straightnessDict - A dictionary with the nth disjoint graph as the key containing a dictionary
                                    with key as the segment index (start node, end node) and
                                    value = straightness of the segment

        Graph.branchPointsDict - A dictionary with the nth disjoint graph as the key and the list of
                                    branch points as the value

        Graph.endPointsDict - A dictionary with the nth disjoint graph as the key and the list of
                                    end points as the value

        Graph.countBranchPointsDict - A dictionary with the nth disjoint graph as the key and the number of
                                        branch points as the value

        Graph.countEndPointsDict -  A dictionary with the nth disjoint graph as the key and the number of
                                        end points as the value

        Graph.degreeDict - A dictionary with the nth disjoint graph as the key containing a dictionary
                                with key as the segment index (start node, end node) and value = segment branching angle
    """
    def __init__(self, segmentation, skeleton, networkxGraph, pixelDimensions, pruningScale, lengthLimit):
        self.networkxGraph = networkxGraph
        self.pixelDims = pixelDimensions
        self.prunScale = pruningScale
        self.lengthLim = lengthLimit
        self.segmentsTotal = 0
        self.segmentsDict = defaultdict(dict)
        self.lengthDict = defaultdict(dict)
        self.volumeDict = defaultdict(dict)
        self.diameterDict = defaultdict(dict)
        self.sumLengthDict = {}
        self.straightnessDict = defaultdict(dict)
        self.branchPointsDict = {}
        self.endPointsDict = {}
        self.countBranchPointsDict = {}
        self.countEndPointsDict = {}
        self.degreeDict = defaultdict(dict)
        self.compTime = 0
        self.postProcessTime = 0

        # calculate distance transform matrix
        self.initTime = time.time()
        self.distTransf = ndi.distance_transform_edt(segmentation, sampling=self.pixelDims)
        self.radiusMatrix = self.distTransf * skeleton
        print("time taken for distance transform is %0.3f seconds" % (time.time() - self.initTime))

        self._prune()  # pruning: delete branches with length below the distance to its closest border
        #self.filaments = list(nx.connected_component_subgraphs(self.networkxGraph))
        self.filaments = list(self.connected_component_subgraphs(self.networkxGraph))

    def setStats(self):
        """
            Set the statistics of a networkx graph
            1) go through each subgraph
            2) calculate all end points and set one end point as beginning point
            3) extract an adjacency dictionary of networkx subgraph
            4) calculate the subgraphs statistics with the class Filament
            5) save statistics for each filament

            Parameters
            ----------
            Graph : networkx graph
                networkx graph of a skeleton
        """
        startTime = time.time()
        countDisjointGraphs = len(self.filaments)
        for ithDisjointGraph, subGraphSkeleton in enumerate(self.filaments):
            # nodeDegreeDict = nx.degree(subGraphSkeleton)
            nodeDegreeDict = dict(nx.degree(subGraphSkeleton))
            endPoints = [k for (k, v) in nodeDegreeDict.items() if v == 1]
            #endPoints.sort()
            if endPoints:
                start = endPoints[0]  # take random end point as beginning
                # print("startPoint = ", start)
                adjacencyDict = nx.to_dict_of_lists(subGraphSkeleton)
                filament = fil.Filament(adjacencyDict, start, self.radiusMatrix, self.pixelDims, self.lengthLim)
                # traversal = filament.dfs_iterative()
                filament.dfs_iterative()
                self.segmentsDict[ithDisjointGraph] = filament.segmentsDict
                # filament may have no segments left after postprocessing
                if len(self.segmentsDict[ithDisjointGraph]) != 0:
                    self.segmentsTotal = self.segmentsTotal + len(self.segmentsDict[ithDisjointGraph])
                    self.lengthDict[ithDisjointGraph] = filament.lengthDict
                    self.sumLengthDict[ithDisjointGraph] = sum(filament.lengthDict.values())
                    self.straightnessDict[ithDisjointGraph] = filament.straightnessDict
                    self.volumeDict[ithDisjointGraph] = filament.volumeDict
                    self.diameterDict[ithDisjointGraph] = filament.diameterDict
                    self.degreeDict[ithDisjointGraph] = filament.degreeDict
                    #self.branchPointsDict[ithDisjointGraph] = filament.brPtsList
                    self.branchPointsDict[ithDisjointGraph] = filament.brPtsDict
                    self.endPointsDict[ithDisjointGraph] = filament.endPtsList
                    #self.countBranchPointsDict[ithDisjointGraph] = len(filament.brPtsList)
                    self.countBranchPointsDict[ithDisjointGraph] = len(filament.brPtsDict)
                    self.countEndPointsDict[ithDisjointGraph] = len(filament.endPtsList)
                    self.compTime += filament.compTime
                    self.postProcessTime += filament.postprocessTime
                else:
                    countDisjointGraphs = countDisjointGraphs-1
        print("summed time taken for dfs calculations is %0.3f seconds" % (self.compTime))
        print("summed time taken for postprocessing is %0.3f seconds" % (self.postProcessTime))
        print("time taken for statistic calculation is %0.3f seconds" % (time.time() - startTime))
        print("filaments=", countDisjointGraphs)
        print("segments = ", self.segmentsTotal)
        # print("segmentsDictTotal = ", self.segmentsDict)

    def _prune(self):
        startTime = time.time()
        #nodeDegreeDict = nx.degree(self.networkxGraph)
        nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        endPtsList = [k for (k, v) in nodeDegreeDict.items() if v == 1]
        branchesToRemove = []
        # find branch for all end points to its closest branch point
        # if distance is below radius * scale => branch to remove
        for endPt in endPtsList:
            visited, stack = set(), [endPt]
            branch = []
            while stack:
                vertex = stack.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    branch.append(vertex)
                    neighbors = [n for n in self.networkxGraph.neighbors(vertex)]
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            stack.append(neighbor)
                    if len(neighbors) > 2:  # branch point found
                        if ms.getLength(branch, self.pixelDims) <= self.radiusMatrix[vertex] * self.prunScale:
                            branchesToRemove.append(branch)
                        break
        # delete branches from adjacency matrix besides branch points
        for branch in branchesToRemove:
            for node in branch[:-1]:
                self.networkxGraph.remove_node(node)
        print("pruned " + str(len(branchesToRemove)) + " branches in %0.3f seconds" % (time.time() - startTime))

    def connected_component_subgraphs(self, G):
        for c in nx.connected_components(G):
            yield G.subgraph(c)


