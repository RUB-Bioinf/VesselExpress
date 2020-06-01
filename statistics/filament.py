import numpy as np
import time

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
    def __init__(self, graph, start):
        self.graph = graph
        self.start = start
        self.endPtsList = []
        self.brPtsList = []
        self.segmentsDict = {}
        self.lengthDict = {}
        self.straightnessDict = {}
        self.degreeDict = {}
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
                            self._predDict[neighbor] = vertex
                            segment = self._getSegment(neighbor)
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

    def _getBranchingDegree(self, segment):
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
        segPred = self._predDict[segment[0]]
        vect1 = [j - i for i, j in zip(segPred, segment[0])]  # requires that brPt is first element
        vect1 = [a * b for a, b in zip(vect1, DIM)]  # multiply pixel length according to dimension
        vect2 = [j - i for i, j in zip(segment[0], segment[1])]
        vect2 = [a * b for a, b in zip(vect2, DIM)]  # multiply pixel length according to dimension
        angle = np.arccos(np.dot(vect1, vect2) / (np.linalg.norm(vect1) * np.linalg.norm(vect2)))
        degree = round(np.degrees(angle), 4)
        return degree

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
        vect = [a * b for a, b in zip(vect, DIM)]  # multiply pixel length with original length
        curveDisplacement = np.linalg.norm(vect)
        self.segmentsDict[segment[0], segment[len(segment) - 1]] = segment
        self.lengthDict[segment[0], segment[len(segment) - 1]] = segLength
        self.straightnessDict[segment[0], segment[len(segment) - 1]] = curveDisplacement / segLength
        if self.start not in segment:
            self.degreeDict[segment[0], segment[len(segment) - 1]] = self._getBranchingDegree(segment)
