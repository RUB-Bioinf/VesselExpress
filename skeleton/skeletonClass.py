import os

import numpy as np
from scipy import ndimage

from skeleton.io_tools import loadStack, saveStack
from metrics.segmentStats import SegmentStats
from skeleton.networkx_graph_from_array import get_networkx_graph_from_array
# NOTE This does the pyx compilation of this extension
import pyximport; pyximport.install() # NOQA
from skeleton.thinVolume import get_thinned
from skeleton.pruning import getPrunedSkeleton

"""
abstract class that encompasses all stages of skeletonization leading to quantification
    1) thinning
    2) pruning
    3) graph conversion
"""


class Skeleton:
    def __init__(self, path, **kwargs):
        # initialize input array
        # path : can be an 3D binary array or a numpy(.npy) array
        # if path is a 3D volume saveSkeletonStack, saves series of
        # skeleton pngs in present directory
        if type(path) is str:
            if path.endswith("npy"):
                # extract rootDir of path
                self.path = os.path.split(path)[0] + os.sep
                self.inputStack = np.load(path)
            else:
                self.path = path
                self.inputStack = loadStack(self.path).astype(bool)
        else:
            self.path = os.getcwd()
            self.inputStack = path
        if kwargs != {}:
            aspectRatio = kwargs["aspectRatio"]
            self.inputStack = ndimage.interpolation.zoom(self.inputStack, zoom=aspectRatio, order=2, prefilter=False)

    def setThinningOutput(self, mode="reflect"):
        # Thinning output
        self.skeletonStack = get_thinned(self.inputStack, mode)

    def setNetworkGraph(self, findSkeleton=False):
        # Network graph of the crowded region removed output
        # Generally the function expects a skeleton
        # and findSkeleton is False by default
        if findSkeleton is True:
            self.setThinningOutput()
        else:
            self.skeletonStack = self.inputStack
        self.graph = get_networkx_graph_from_array(self.skeletonStack)

    def setPrunedSkeletonOutput(self):
        # Prune unnecessary segments in crowded regions removed skeleton
        self.setNetworkGraph(findSkeleton=True)
        self.outputStack = getPrunedSkeleton(self.skeletonStack, self.graph)

    def getNetworkGraph(self):
        # Network graph of the final output skeleton stack
        self.setPrunedSkeletonOutput()
        self.outputGraph = get_networkx_graph_from_array(self.outputStack)

    def saveSkeletonStack(self):
        # Save output skeletonized stack as series of pngs in the path under a subdirectory skeleton
        # in the input "path"
        self.setPrunedSkeletonOutput()
        saveStack(self.outputStack, self.path + "skeleton/")

    def getSegmentStatsBeforePruning(self):
        # stats before pruning the braches
        self.setNetworkGraph()
        self.statsBefore = SegmentStats(self.graph)
        self.statsBefore.setStats()

    def setSegmentStatsAfterPruning(self):
        # stats after pruning the braches
        self.getNetworkGraph()
        self.statsAfter = SegmentStats(self.outputGraph)
        self.statsAfter.setStats()
