import numpy as np
import time
import math

# pixel dimensions in microns with [z, y, x]
DIM = [2.0, 1.015625, 1.015625]
# DIM = [1, 1]


class Filament:
    """
        Find statistics on a connected graph of a skeleton

        Parameters
        ----------
        graph : dictionary of lists (adjacency list) with key as node and value = list with its neighboring nodes

        start : list of coordinates
            beginning point for DFS (must be an end point)
            3D: [z, y, x]   2D: [y, x]

        Examples
        --------
        Filament.endPtsList - A list containing all nodes with only one other node connected to them

        Filament.branchPtsList - A list containing all nodes with more than 2 nodes connected to them

        Filament.segmentsDict - A dictionary containing all segments (path from branch/end point to branch/end point)
            with key as segment index (start node, end node) and value = list of nodes in segment

        Filament.lengthDict - A dictionary with key as segment index (start node, end node) and
            value = length of the segment

        Filament.straightnessDict - A dictionary with key as segment index (start node, end node) and
            value = straightness of the segment

        Filament.degreeDict - A dictionary with key as segment index (start node, end node) and
            value = segment branching angle
    """
    def __init__(self, graph, start, skelRadii):
        self.graph = graph
        self.start = start
        self.skelRadii = skelRadii
        self.endPtsList = []
        self.brPtsList = []
        self.segmentsDict = {}
        self.lengthDict = {}
        self.straightnessDict = {}
        self.degreeDict = {}
        self.volumeDict = {}
        self.diameterDict = {}
        self._predDict = {}
        self.compTime = 0

    def dfs_iterative(self):
        """
            Iterate over the graph in a depth-first-search. If a branch or end point is found, retrieve the segment
            and calculate its statistics
        """
        startTime = time.time()
        visited, stack = set(), [self.start]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        self._predDict[neighbor] = vertex
                        stack.append(neighbor)
                    elif neighbor in visited and not self._predDict[vertex] == neighbor:  # cycle found
                        if len(self.graph[neighbor]) > 2:  # neighbor is branch point
                            oldPred = self._predDict[neighbor]
                            self._predDict[neighbor] = vertex  # change predecessor to get segment of cycle
                            segment = self._getSegment(neighbor)
                            self._predDict[neighbor] = oldPred  # change back to old predecessor
                            self._setSegStats(segment)
                if len(self.graph[vertex]) == 1:  # end point found
                    self.endPtsList.append(vertex)
                    if vertex != self.start:
                        segment = self._getSegment(vertex)
                        self._setSegStats(segment)
                elif len(self.graph[vertex]) > 2:  # branch point found
                    self.brPtsList.append(vertex)
                    segment = self._getSegment(vertex)
                    self._setSegStats(segment)
        self.compTime = time.time() - startTime

    def _getSegment(self, node):
        """
            Find the segment of a branch or end node by iterating over its predecessors

            Parameters
            ----------
            node : list of node coordinates
                3D: [z, y, x]   2D: [y, x]

            Returns
            -------
            segmentList : list of nodes in the segment
        """
        segmentList = [node]
        while True:
            node = self._predDict[node]
            segmentList.insert(0, node)
            if len(self.graph[node]) == 1 or len(self.graph[node]) > 2:
                break
        return segmentList

    def _getLength(self, path, dimensions=DIM):
        """
            Find length of a path as distance between nodes in it

            Parameters
            ----------
            path : list
                list of nodes in the path

            dimensions : list
                list with pixel dimensions in desired unit (e.g. microns)
                3D: [z, y, x]   2D: [y, x]

            Returns
            -------
            length : float
                Length of path
        """
        length = 0
        for index, item in enumerate(path):
            if index + 1 != len(path):
                item2 = path[index + 1]
            vect = [j - i for i, j in zip(item, item2)]
            vect = [a * b for a, b in zip(vect, dimensions)]  # multiply pixel length with original length
            length += np.linalg.norm(vect)
        return length

    def _getBranchingDegree(self, segment, factor=0.25):
        """
            Find branching angle of a segment

            Parameters
            ----------
            segment : list
                list of nodes in the segment

            Returns
            -------
            degree : float
                Degree with 4 decimal places
        """
        segPredList = self._getSegment(segment[0])
        segPredList.reverse()
        # take neighboring points from branching point
        if factor <= 0:
            segPredPt = segPredList[1]
            segPt = segment[1]
        # take half of the segments from branching point
        elif factor == 0.5:
            segPredPt = segPredList[int(len(segPredList) * factor)]
            segPt = segment[int(len(segment) * factor)]
        # take the whole segment from branching point
        elif factor >= 1:
            segPredPt = segPredList[len(segPredList) - 1]
            if segment[len(segment) - 1] == segment[0]:  # in case of a circle, take pre-last point
                segPt = segment[len(segment) - 2]
            else:
                segPt = segment[len(segment) - 1]
        # take points according to the factor of the segment lengths
        else:
            # determine point of predecessor
            if round(len(segPredList) * factor) == 0 or len(segPredList) == 2:
                segPredPt = segPredList[1]
            else:
                segPredPt = segPredList[round(len(segPredList) * factor)]
            # determine point of segment
            if round(len(segment) * factor) == 0 or len(segment) == 2:
                segPt = segment[1]
            else:
                if segment[round(len(segment) * factor)] == segment[0]:  # in case of a circle, take pre-last point
                    segPt = segment[len(segment) - 2]
                else:
                    segPt = segment[round(len(segment) * factor)]
        segPredVect = [j - i for i, j in zip(segPredPt, segment[0])]
        segPredVect = [a * b for a, b in zip(segPredVect, DIM)]
        segVect = [j - i for i, j in zip(segment[0], segPt)]
        segVect = [a * b for a, b in zip(segVect, DIM)]
        cosine_angle = np.dot(segPredVect, segVect) / (np.linalg.norm(segPredVect) * np.linalg.norm(segVect))
        angle = np.arccos(round(cosine_angle, 4))
        return round(np.degrees(angle), 4)

    def _getVolume(self, segment):
        """
            Calculate volume and average diameter of a segment

            Parameters
            ----------
            segment : list
                list of nodes in the segment

            Returns
            -------
            volume, average diameter : float
        """
        volume = 0
        diameter = 0
        for skelPt in segment:
            volume = volume + math.pi * self.skelRadii[skelPt]**2
            diameter = diameter + self.skelRadii[skelPt] * 2
        return volume, diameter / len(segment)

    def _setSegStats(self, segment):
        """
            Sets the statistics for a segment

            Parameters
            ----------
            segment : list
                list of nodes in the segment
        """
        segLength = self._getLength(segment)
        vect = [j - i for i, j in zip(segment[0], segment[len(segment) - 1])]
        vect = [a * b for a, b in zip(vect, DIM)]  # multiply pixel length with pixel dimension
        curveDisplacement = np.linalg.norm(vect)
        self.segmentsDict[segment[0], segment[len(segment) - 1]] = segment
        self.lengthDict[segment[0], segment[len(segment) - 1]] = segLength
        self.straightnessDict[segment[0], segment[len(segment) - 1]] = curveDisplacement / segLength
        volumeDiameter = self._getVolume(segment)
        self.volumeDict[segment[0], segment[len(segment) - 1]] = volumeDiameter[0]
        self.diameterDict[segment[0], segment[len(segment) - 1]] = volumeDiameter[1]
        if self.start not in segment:  # if start point is included in segment its either a line or has no predecessor
            self.degreeDict[segment[0], segment[len(segment) - 1]] = self._getBranchingDegree(segment)

    def _postprocess(self, lowLim=2.5):
        # find segments in lengthDict which are below the limit
        keysToRemoveList = []
        for segKey in self.lengthDict:
            if self.lengthDict[segKey] <= lowLim:
                keysToRemoveList.append(segKey)
        # delete those segments from dictionarys
        for key in keysToRemoveList:
            del self.segmentsDict[key]
            del self.lengthDict[key]
            del self.straightnessDict[key]
            if key in self.degreeDict:
                del self.degreeDict[key]