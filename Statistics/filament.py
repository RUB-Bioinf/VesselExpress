import numpy as np
import time
import Statistics.measurements as ms

class Filament:
    """
        Find statistics on a graph of a skeleton

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

        Notes
        --------
        straightness = curveDisplacement / curveLength
        """
    def __init__(self, graph, start, skelRadii, pixelDimensions, lengthLimit):
        self.graph = graph
        self.start = start
        self.skelRadii = skelRadii
        self.pixelDims = pixelDimensions
        self.lengthLim = lengthLimit
        self.endPtsList = []
        # self.brPtsList = []
        self.brPtsDict = {}
        self.segmentsDict = {}
        self.lengthDict = {}
        self.straightnessDict = {}
        self.degreeDict = {}
        self.volumeDict = {}
        self.diameterDict = {}
        self._predDict = {}
        self.compTime = 0
        self.postprocessTime = 0

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
                            self._predDict[neighbor] = vertex   # change predecessor to get segment of cycle
                            segment = self._getSegment(neighbor)
                            self._predDict[neighbor] = oldPred  # change back to old predecessor
                            self._setSegStats(segment)
                if len(self.graph[vertex]) == 1:    # end point found
                    self.endPtsList.append(vertex)
                    if vertex != self.start:
                        segment = self._getSegment(vertex)
                        self._setSegStats(segment)
                elif len(self.graph[vertex]) > 2:   # branch point found
                    # self.brPtsList.append(vertex)
                    self.brPtsDict[vertex] = len(self.graph[vertex])
                    segment = self._getSegment(vertex)
                    self._setSegStats(segment)
        self.compTime = time.time() - startTime
        postprocStart = time.time()
        self._postprocess()
        self.postprocessTime = time.time() - postprocStart

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
            node = self._predDict.get(node)
            if node is None:  # may happen due to postprocessing removing predecessors of old branching points
                return None
            segmentList.insert(0, node)
            if len(self.graph[node]) == 1 or len(self.graph[node]) > 2:
                break
        return segmentList

    def _getBranchingDegree(self, segment, factor=0.25):
        segPredList = self._getSegment(segment[0])
        if segPredList is None:
            return None
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
            segPredPt = segPredList[len(segPredList)-1]
            if segment[len(segment) - 1] == segment[0]:     # in case of a circle, take pre-last point
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
        segPredVect = [a * b for a, b in zip(segPredVect, self.pixelDims)]
        segVect = [j - i for i, j in zip(segment[0], segPt)]
        segVect = [a * b for a, b in zip(segVect, self.pixelDims)]

        # angle = np.arccos(np.dot(segPredVect, segVect) / (np.linalg.norm(segPredVect) * np.linalg.norm(segVect)))
        cosine_angle = np.dot(segPredVect, segVect) / (np.linalg.norm(segPredVect) * np.linalg.norm(segVect))
        angle = np.arccos(round(cosine_angle, 4))
        #print("-----------------")
        #print("segment=", segment)
        #print("segmentLength=", len(segment))
        # print("segmentPred=", segPredList)
        # #print("segPredIndex=", (len(segPredList) - (round(len(segPredList) * factor))))
        # #print("segPredPt=", segPredPt)
        #print("segmentPredVectPts=", segPredPt, segment[0])
        # print("segPredVect=", segPredVect)
        # #print("segPt=", segPt)
        # print("segPtIndex=", (len(segment) * factor))
        #print("segmentVectPts=", segPt, segment[0])
        # print("segVect=", segVect)
        #print("degree=", degree)
        return round(np.degrees(angle), 4)

    def _setSegStats(self, segment):
        """
            Sets the statistics for a segment

            Parameters
            ----------
            segment : list
                list of nodes in the segment
        """
        segLength = ms.getLength(segment, self.pixelDims)
        vect = [j - i for i, j in zip(segment[0], segment[len(segment) - 1])]
        vect = [a * b for a, b in zip(vect, self.pixelDims)]  # multiply pixel length with pixel dimension
        curveDisplacement = np.linalg.norm(vect)
        self.segmentsDict[segment[0], segment[len(segment) - 1]] = segment
        self.lengthDict[segment[0], segment[len(segment) - 1]] = segLength
        self.straightnessDict[segment[0], segment[len(segment) - 1]] = curveDisplacement / segLength
        volumeDiameter = ms.getVolume(self.skelRadii, segment)
        self.volumeDict[segment[0], segment[len(segment) - 1]] = volumeDiameter[0]
        self.diameterDict[segment[0], segment[len(segment) - 1]] = volumeDiameter[1]
        # self.volumeDict[segment[0], segment[len(segment) - 1]] = self._getVolume(segment)
        if self.start not in segment:   # if start point is included in segment its either a line or has no predecessor
            self.degreeDict[segment[0], segment[len(segment) - 1]] = self._getBranchingDegree(segment)

    def _deletePath(self, path):
        for i in range(len(path) - 1):
            self._removeEdge(path[i], path[i + 1])

    def _removeEdge(self, u, v):
        self.graph.get(u).remove(v)
        self.graph.get(v).remove(u)

    def _postprocess(self):
        # find segments in lengthDict which are below the limit
        keysToRemoveList = []
        brPtCandidates = set()
        for segKey in self.lengthDict:
            if self.lengthDict[segKey] <= self.lengthLim:
                keysToRemoveList.append(segKey)
        # delete those segments from dictionaries
        for key in keysToRemoveList:
            self._deletePath(self.segmentsDict[key])
            del self.segmentsDict[key]
            del self.lengthDict[key]
            del self.straightnessDict[key]
            del self.volumeDict[key]
            del self.diameterDict[key]
            if key in self.degreeDict:
                del self.degreeDict[key]
            if key[0] in self.endPtsList:
                self.endPtsList.remove(key[0])
            if key[1] in self.endPtsList:
                self.endPtsList.remove(key[1])
            # add branch points to possible deletable candidates
            if key[0] in self.brPtsDict:
                brPtCandidates.add(key[0])
            if key[1] in self.brPtsDict:
                brPtCandidates.add(key[1])
        # check if branch points still remain branch points after postprocessing
        for brPt in brPtCandidates:
            # branch point becomes normal point connecting two segments together
            if len(self.graph[brPt]) == 2:
                #print("brPt ", brPt, " becomes normal point with neighbors=", self.graph[brPt])
                del self.brPtsDict[brPt]
                # find both segments connected by the branch pt
                segments = [v for k, v in self.segmentsDict.items() if k[0] == brPt or k[1] == brPt]
                # delete old segments from segment dictionaries (either one segment if circle otherwise 2 segments)
                #print(segments)
                if len(segments) > 1:
                    segKey1 = (segments[0][0], segments[0][-1])
                    del self.segmentsDict[segKey1]
                    del self.lengthDict[segKey1]
                    del self.straightnessDict[segKey1]
                    del self.volumeDict[segKey1]
                    del self.diameterDict[segKey1]
                    if len(segments) != 1:  # if segment is not a circle delete second segment from dictionaries
                        segKey2 = (segments[1][0], segments[1][-1])
                        del self.segmentsDict[segKey2]
                        del self.lengthDict[segKey2]
                        del self.straightnessDict[segKey2]
                        del self.volumeDict[segKey2]
                        del self.diameterDict[segKey2]
                        # combine both segments to one segment and calculate its statistics
                        if segments[0][-1] == brPt and segments[1][0] == brPt:
                            combSegments = segments[0] + segments[1][1:]
                        elif segments[0][0] == brPt and segments[1][-1] == brPt:
                            combSegments = segments[1] + segments[0][1:]
                        # in this case the predecessor of the brPt has been deleted
                        elif segments[0][0] == brPt and segments[1][0] == brPt:
                            combSegments = segments[0][::-1]+ segments[1][1:]
                        elif segments[0][-1] == brPt and segments[1][-1] == brPt:
                            combSegments = segments[0][:-1] + segments[1][::-1]
                        self._setSegStats(combSegments)
            # branch point becomes end point
            elif len(self.graph[brPt]) == 1:
                # print("brPt ", brPt, " becomes endPoint")
                self.endPtsList.append(brPt)
            # all branches of a branch point were removed => delete branch point from dict
            elif len(self.graph[brPt]) == 0:
                # print("brPt", brPt, " becomes single point")
                del self.brPtsDict[brPt]


