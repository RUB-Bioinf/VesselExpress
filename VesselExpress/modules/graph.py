import os

import networkx as nx
from collections import defaultdict
import time

import numpy as np

import filament as fil
from scipy.ndimage import distance_transform_edt
import csv
import dask.array as da
# from dask.distributed import Client

import measurements as ms


class distance_transform_edt_dask:
    def __init__(self, sampling):
        self.sampling = sampling

    def compute_distance_transform(self, im):
        return distance_transform_edt(im.astype(float), sampling=self.sampling)


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
    def __init__(self, segmentation, skeleton, networkxGraph, pixelDimensions, pruningScale, lengthLimit, diaScale,
                 branchingThreshold, expFlag, smallRAMmode, infoFile, graphCreation, fileName, removeBorderEndPts,
                 removeEndPtsFromSmallFilaments, interpolate, splineDegree, cut_neighbor_brpt_segs):
        self.skeleton = skeleton
        self.networkxGraph = networkxGraph
        self.pixelDims = pixelDimensions
        self.prunScale = pruningScale
        self.lengthLim = lengthLimit
        self.diaScale = diaScale
        self.branchingThresh = branchingThreshold
        self.infoFile = infoFile
        self.graphCreation = graphCreation
        self.fileName = fileName
        self.removeBorderEndPts = removeBorderEndPts
        self.removeEndPtsFromSmallFilaments = removeEndPtsFromSmallFilaments
        self.interpolate = interpolate
        self.splineDegree = splineDegree
        self.cut_neighbor_brpt_segs = cut_neighbor_brpt_segs
        self.expFlag = expFlag
        self.smallRAMmode = smallRAMmode
        self.segmentsDict = defaultdict(dict)
        self.countSegmentsDict = {}
        self.branchPointsDict = {}
        self.endPointsDict = {}
        self.countBranchPointsDict = {}
        self.countEndPointsDict = {}
        self.compTime = 0
        self.postProcessTime = 0
        self.endPtsTopVsBottom = 0
        self.nodesFinal = set()
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
        self.nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        # dictionaries containing all filament, segment and branch point statistics
        self.segStatsDict = defaultdict(dict)
        self.filStatsDict = defaultdict(dict)
        self.branchesBrPtDict = defaultdict(dict)

        # calculate distance transform matrix
        self.initTime = time.time()
        if self.smallRAMmode == 0:
            self.distTransf = distance_transform_edt(segmentation, sampling=self.pixelDims)
        else:
            im_dask = da.from_array(segmentation, chunks=(128, 128, 128))
            if segmentation.shape[0] * segmentation.shape[1] * segmentation.shape[2] > 16777216:
                im_dask = da.from_array(segmentation, chunks='auto')
            edt_func = distance_transform_edt_dask(
                sampling=self.pixelDims
            )
            self.distTransf = da.map_overlap(
                edt_func.compute_distance_transform,
                im_dask,
                dtype="float",
                depth=15,
                boundary='reflect'
            )
        self.radiusMatrix = self.distTransf * self.skeleton
        self.runTimeDict['distTransformation'] = round(time.time() - self.initTime, 3)

        # pruning: delete branches with length below the distance to its closest border
        if self.smallRAMmode == 1:
            # client = Client()
            self.radiusMatrix.to_zarr('tmp_zarr' + os.sep + self.fileName + '_radiusMatrix.zarr')
            self._prune(da.from_zarr('tmp_zarr' + os.sep + self.fileName + '_radiusMatrix.zarr'))
            #client.submit(self._prune, da.from_zarr('tmp_zarr' + os.sep + self.fileName + '_radiusMatrix.zarr'))
        else:
            self._prune(self.radiusMatrix)

        self.filaments = list(self.connected_component_subgraphs(self.networkxGraph))
        # print("found {} filaments".format(len(self.filaments)))

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
                                        self.diaScale, self.branchingThresh, self.expFlag, self.smallRAMmode,
                                        self.fileName, self.removeBorderEndPts, self.removeEndPtsFromSmallFilaments,
                                        self.interpolate, self.splineDegree, self.cut_neighbor_brpt_segs)
                filament.dfs_iterative()
                self.segmentsDict[ithDisjointGraph] = filament.segmentsDict

                # save nodes of filament after processing
                nodes = list(self.segmentsDict[ithDisjointGraph].values())
                nodes = set([item for sublist in nodes for item in sublist])
                self.nodesFinal.update(nodes)

                # filament may have no segments left after postprocessing
                if len(self.segmentsDict[ithDisjointGraph]) != 0:
                    self.countSegmentsDict[ithDisjointGraph] = len(self.segmentsDict[ithDisjointGraph])
                    self.branchPointsDict[ithDisjointGraph] = filament.brPtsDict
                    self.endPointsDict[ithDisjointGraph] = filament.endPtsList
                    self.countBranchPointsDict[ithDisjointGraph] = len(filament.brPtsDict)
                    self.countEndPointsDict[ithDisjointGraph] = len(filament.endPtsList)
                    self.infoDict['segments'] += len(self.segmentsDict[ithDisjointGraph])
                    self.runTimeDict['dfsComp'] += filament.compTime
                    self.runTimeDict['postProcessing'] += filament.postprocessTime
                    self.infoDict['postProcBranches'] += filament.postprocBranches
                    self.infoDict['postProcEndPts'] += filament.postprocEndPts

                    # fill dictionaries containing all filament, segment and branch point statistics
                    self.segStatsDict[ithDisjointGraph] = filament.segmentStats
                    self.filStatsDict[ithDisjointGraph]['TerminalPoints'] = self.countEndPointsDict[ithDisjointGraph]
                    self.filStatsDict[ithDisjointGraph]['BranchPoints'] = self.countBranchPointsDict[ithDisjointGraph]
                    self.filStatsDict[ithDisjointGraph]['Segments'] = self.countSegmentsDict[ithDisjointGraph]
                    self.branchesBrPtDict[ithDisjointGraph] = filament.brPtsDict
                else:
                    self.infoDict['filaments'] -= 1
        self.runTimeDict['statCalculation'] = round(time.time() - startTime, 3)

        if self.graphCreation == 1:
            # remove nodes which were removed in postprocessing of filaments in the overall graph
            nodes_graph = set(self.networkxGraph.nodes)
            nodes_to_remove = nodes_graph - self.nodesFinal
            for node in nodes_to_remove:
                self.networkxGraph.remove_node(node)
            self.skeleton = self._get_final_skeleton()

        if self.expFlag == 1:
            self.endPtsTopVsBottom = self.top_endPts_vs_bottom_endPts()

        if self.infoFile:
            self._writeInfoFile()

    def _get_final_skeleton(self):
        skel = np.zeros(self.skeleton.shape)
        for ind in list(self.networkxGraph.nodes):
            skel[ind] = 1
        return skel

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

    def _prune(self, radiusMat):
        """
            This implements the pruning as described by Montero & Lang
            Skeleton pruning by contour approximation and the integer medial axis transform.
            Computers & Graphics 36, 477-487 (2012).
            Branches are removed when |ep - bp|^2 <= s * |f - bp|^2
            where ep = end point, bp = branch point, s = scaling factor, f = closest boundary point
        """
        startTime = time.time()
        endPtsList = [k for (k, v) in self.nodeDegreeDict.items() if v == 1]
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
                        if ms.getLength(branch, self.pixelDims) <= radiusMat[vertex] * self.prunScale:
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
