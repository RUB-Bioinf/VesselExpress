import networkx as nx
from collections import defaultdict
import time
import statistics.filament as fil


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

        Graph.lengthDict - A dictionary with the nth disjoint graph as the key containing a dictionary
                                with key as the segment index (start node, end node) and value = length of the segment

        Graph.sumLengthDict - A dictionary with the nth disjoint graph as the key and the total length as the value

        Graph.straightnessDict - A dictionary with the nth disjoint graph as the key containing a dictionary
                                    with key as the segment index (start node, end node) and
                                    value = straightness of the segment

        Graph.countBranchPointsDict - A dictionary with the nth disjoint graph as the key and the number of
                                        branch points as the value

        Graph.countEndPointsDict -  A dictionary with the nth disjoint graph as the key and the number of
                                        end points as the value

        Graph.degreeDict - A dictionary with the nth disjoint graph as the key containing a dictionary
                                with key as the segment index (start node, end node) and value = segment branching angle
    """
    def __init__(self, networkxGraph):
        self.networkxGraph = networkxGraph
        self.segmentsTotal = 0
        self.segmentsDict = defaultdict(dict)
        self.lengthDict = defaultdict(dict)
        self.sumLengthDict = {}
        self.straightnessDict = defaultdict(dict)
        self.countBranchPointsDict = {}
        self.countEndPointsDict = {}
        self.degreeDict = defaultdict(dict)
        self.filaments = list(nx.connected_component_subgraphs(self.networkxGraph))
        self.compTime = 0

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
        for ithDisjointGraph, subGraphSkeleton in enumerate(self.filaments):
            nodeDegreeDict = nx.degree(subGraphSkeleton)
            endPoints = [k for (k, v) in nodeDegreeDict.items() if v == 1]
            endPoints.sort()
            if endPoints:
                start = endPoints[0]  # take random end point as beginning
                adjacencyDict = nx.to_dict_of_lists(subGraphSkeleton)
                filament = fil.Filament(adjacencyDict, start)
                filament.dfs_iterative()
                self.segmentsDict[ithDisjointGraph] = filament.segmentsDict
                self.segmentsTotal = self.segmentsTotal + len(self.segmentsDict[ithDisjointGraph])
                self.lengthDict[ithDisjointGraph] = filament.lengthDict
                self.sumLengthDict[ithDisjointGraph] = sum(filament.lengthDict.values())
                self.straightnessDict[ithDisjointGraph] = filament.straightnessDict
                self.degreeDict[ithDisjointGraph] = filament.degreeDict
                self.countBranchPointsDict[ithDisjointGraph] = len(filament.brPtsList)
                self.countEndPointsDict[ithDisjointGraph] = len(filament.endPtsList)
                self.compTime += filament.compTime
        print("summed time taken for dfs calculations is %0.3f seconds" % (self.compTime))
        print("total time for statistic calculation is %0.3f seconds" % (time.time() - startTime))



