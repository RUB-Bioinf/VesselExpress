import networkx as nx
from collections import defaultdict
import time
import filament as fil
from scipy.ndimage import distance_transform_edt
import csv
import dask.array as da

import measurements as ms


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

        Graph.volumeDict - A dictionary with the nth disjoint graph as the key containing a dictionary
                                with key as the segment index (start node, end node) and value = volume of the segment

        Graph.diameterDict - A dictionary with the nth disjoint graph as the key containing a dictionary
                                with key as the segment index (start node, end node) and value = avg diameter of the segment
    """
    def __init__(self, segmentation, skeleton, networkxGraph, pixelDimensions, pruningScale, lengthLimit,
                 branchingThreshold, expFlag, infoFile):
        self.networkxGraph = networkxGraph
        self.pixelDims = pixelDimensions
        self.prunScale = pruningScale
        self.lengthLim = lengthLimit
        self.branchingThresh = branchingThreshold
        self.infoFile = infoFile
        self.expFlag = expFlag
        self.segmentsDict = defaultdict(dict)
        self.countSegmentsDict = {}
        self.branchPointsDict = {}
        self.endPointsDict = {}
        self.countBranchPointsDict = {}
        self.countEndPointsDict = {}
        self.compTime = 0
        self.postProcessTime = 0
        self.endPtsTopVsBottom = 0
        self.runTimeDict = {
            'distTransformation': 0,
            'pruning': 0,
            'dfsComp': 0,
            'postProcessing': 0,
            'statCalculation': 0
        }
        self.infoDict = {
            'pruning': 0,
            'filaments': 0,
            'segments': 0,
            'postProcBranches': 0,
            'postProcEndPts': 0
        }
        # dictionaries containing all filament, segment and branch point statistics
        self.segStatsDict = defaultdict(dict)
        self.filStatsDict = defaultdict(dict)
        self.branchesBrPtDict = defaultdict(dict)

        # calculate distance transform matrix
        self.initTime = time.time()
        self.distTransf = distance_transform_edt(segmentation, sampling=self.pixelDims)
        #im_dask = segmentation.rechunk(chunks='auto')
        # note: this is only for large data, when debugging, please use:
        # im_dask = segmentation.rechunk(chunks=(128, 128, 128))
        # this is because for small data, 'auto' will just keep the whole data as one chunk
        # then it is meaningless for debugging or testing
        # here "depth=15" should be fine for most case, unless we have very thick vessels
        # "depth" controlls how much overlap between each neighboring chunks
        #self.distTransf = da.map_overlap(distance_transform_edt, im_dask, dtype="float", depth=15)
        self.radiusMatrix = self.distTransf * skeleton
        self.runTimeDict['distTransformation'] = round(time.time() - self.initTime, 3)

        # pruning: delete branches with length below the distance to its closest border
        self._prune()

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
        # statistic calculation
        startTime = time.time()
        self.infoDict['filaments'] = len(self.filaments)
        for ithDisjointGraph, subGraphSkeleton in enumerate(self.filaments):
            nodeDegreeDict = dict(nx.degree(subGraphSkeleton))
            endPoints = [k for (k, v) in nodeDegreeDict.items() if v == 1]
            if endPoints:
                start = endPoints[0]  # take random end point as beginning
                adjacencyDict = nx.to_dict_of_lists(subGraphSkeleton)
                filament = fil.Filament(adjacencyDict, start, self.radiusMatrix, self.pixelDims, self.lengthLim,
                                        self.branchingThresh, self.expFlag)
                filament.dfs_iterative()
                self.segmentsDict[ithDisjointGraph] = filament.segmentsDict
                # filament may have no segments left after postprocessing
                if len(self.segmentsDict[ithDisjointGraph]) != 0:
                    self.countSegmentsDict[ithDisjointGraph] = len(self.segmentsDict[ithDisjointGraph])
                    self.branchPointsDict[ithDisjointGraph] = filament.brPtsDict
                    self.endPointsDict[ithDisjointGraph] = filament.endPtsList
                    self.countBranchPointsDict[ithDisjointGraph] = len(filament.brPtsDict)
                    if self.countSegmentsDict[ithDisjointGraph] > 4:
                        self.countEndPointsDict[ithDisjointGraph] = len(filament.endPtsList)
                    else:
                        self.countEndPointsDict[ithDisjointGraph] = 0
                    self.infoDict['segments'] += len(self.segmentsDict[ithDisjointGraph])
                    self.runTimeDict['dfsComp'] += filament.compTime
                    self.runTimeDict['postProcessing'] += filament.postprocessTime
                    self.infoDict['postProcBranches'] += filament.postprocBranches
                    self.infoDict['postProcEndPts'] += filament.postprocEndPts

                    # fill dictionaries containing all filament, segment and branch point statistics
                    self.segStatsDict[ithDisjointGraph] = filament.segmentStats
                    if self.countSegmentsDict[ithDisjointGraph] > 4:
                        self.filStatsDict[ithDisjointGraph]['TerminalPoints'] = self.countEndPointsDict[ithDisjointGraph]
                    else:
                        self.filStatsDict[ithDisjointGraph]['TerminalPoints'] = 0
                    self.filStatsDict[ithDisjointGraph]['BranchPoints'] = self.countBranchPointsDict[ithDisjointGraph]
                    self.filStatsDict[ithDisjointGraph]['Segments'] = self.countSegmentsDict[ithDisjointGraph]
                    self.branchesBrPtDict[ithDisjointGraph] = filament.brPtsDict
                else:
                    self.infoDict['filaments'] -= 1
        self.runTimeDict['statCalculation'] = round(time.time() - startTime, 3)

        if self.expFlag == 1:
            self.endPtsTopVsBottom = self.top_endPts_vs_bottom_endPts()

        if self.infoFile:
            self._writeInfoFile()

    def _writeInfoFile(self):
        with open(self.infoFile, 'w', newline='') as csvfile:
            fieldnames = ['step', 'time(s)', 'comment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({'step': 'Distance transformation', 'time(s)': str(self.runTimeDict['distTransformation'])})
            writer.writerow({'step': 'Pruning', 'time(s)': str(self.runTimeDict['pruning']),
                             'comment': 'pruned ' + str(self.infoDict['pruning']) + ' branches'})
            writer.writerow({'step': 'DFS calculations', 'time(s)': str(round(self.runTimeDict['dfsComp'], 3))})
            writer.writerow({'step': 'Postprocessing', 'time(s)': str(round(self.runTimeDict['postProcessing'], 3)),
                             'comment': 'removed ' + str(self.infoDict['postProcBranches']) + ' branches and ' +
                             str(self.infoDict['postProcEndPts']) + ' end points'})
            writer.writerow({'step': 'Statistic calculation', 'time(s)': str(self.runTimeDict['statCalculation']),
                             'comment': 'filaments=' + str(self.infoDict['filaments']) +
                                        ' segments=' + str(self.infoDict['segments'])})

    def _prune(self):
        """
            This implements the pruning as described by Montero & Lang
            Skeleton pruning by contour approximation and the integer medial axis transform.
            Computers & Graphics 36, 477-487 (2012).
            Branches are removed when |ep - bp|^2 <= s * |f - bp|^2
            where ep = end point, bp = branch point, s = scaling factor, f = closest boundary point
        """
        startTime = time.time()
        nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        endPtsList = [k for (k, v) in nodeDegreeDict.items() if v == 1]
        branchesToRemove = []
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
        self.runTimeDict['pruning'] = round(time.time() - startTime, 3)
        self.infoDict['pruning'] = len(branchesToRemove)

    def connected_component_subgraphs(self, G):
        for c in nx.connected_components(G):
            yield G.subgraph(c)

    def top_endPts_vs_bottom_endPts(self, gap=15):
        z = self.radiusMatrix.shape[0] - 1
        top_endPts = 0
        bottom_endPts = 0
        for filament in self.endPointsDict:
            if self.countSegmentsDict[filament] > 4:
                for endPts in self.endPointsDict[filament]:
                    if endPts[0] < gap:
                        bottom_endPts += 1
                    if endPts[0] >= z-gap:
                        top_endPts += 1
        if bottom_endPts != 0:
            ratio = round(top_endPts/bottom_endPts, 4)
        else:
            ratio = "NULL"
        return ratio
