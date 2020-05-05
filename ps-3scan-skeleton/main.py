import numpy as np
# NOTE This does the pyx compilation of this extension
import pyximport; pyximport.install() # NOQA
import skeleton.thinVolume as thinVol
from skimage import io
from skimage.external import tifffile as tif
import skeleton.networkx_graph_from_array as netGraphArr
import metrics.segmentStats as segStats
from skeleton.pruning import getPrunedSkeleton
import pickle
from runscripts import animation
import networkx as nx
import matplotlib.pyplot as plt
import utils
import os

# Test 2D-Arrays
# Bsp. 1 mit einfachem azyklischem Baum
arr1 = np.array([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1]])
# Bsp. 2 mit zusätzlicher Linie
arr2 = np.array([[0, 0, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1]])
# Bsp. 3 mit Zyklus
arr3 = np.array([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0]])
# Bsp. 4 mit azyklischem und zyklischem Graph
arr4 = np.array([[1, 0, 1, 0, 0, 0],
                 [0, 1, 0, 0, 1, 0],
                 [0, 1, 0, 1, 0, 1],
                 [0, 1, 0, 0, 1, 1]])
# Bsp. 5 mit Linien
arr5 = np.array([[0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1]])
# Bsp. 6 cyclic tree with clique removals
arr6 = np.array([[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0]])
# Bsp. 7 cyclic tree
arr7 = np.array([[0, 1, 0, 1, 0],
                 [1, 0, 1, 0, 1],
                 [0, 1, 0, 1, 1],
                 [1, 0, 1, 0, 1],
                 [0, 1, 0, 0, 1]])
# Bsp. 8 zwei acyclic trees
arr8 = np.array([[1, 0, 1, 0, 0, 1, 0],
                 [0, 1, 0, 0, 0, 1, 0],
                 [1, 0, 1, 0, 0, 1, 0],
                 [1, 0, 1, 0, 1, 0, 1]])
# Bsp. 9 acyclic tree mit 2 BrPts
arr9 = np.array([[1, 0, 1, 1],
                 [0, 1, 0, 0],
                 [0, 1, 0, 0],
                 [1, 0, 1, 1]])
# Test 3D-Array
arr10 = np.array([[[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1]],
                [[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1]]])
# netGraphSkel = netGraphArr.get_networkx_graph_from_array(arr10)
# print("NODES:", netGraphSkel.nodes())
# print("EDGES:", netGraphSkel.edges())

# plt.imshow(arr1)
# plt.show()
# nx.draw(netGraphSkel)
# plt.draw()
# plt.show()

# specify path and image
PATH = '/Users/philippaspangenberg/Documents/TestBilder/49-B5_contra_RB_cmp/'
IMG = 'C2-49slices-B5_contra_RB_cmp'

# 1) read tif image as numpy array
im = io.imread(PATH + IMG + '.tif')                     # reads image as array (z, y, x)
binArr = im/np.max(im)                                  # make binary array of 3D image/ thresholded 3D volume
np.save(PATH + 'THR-' + IMG + '.npy', binArr)           # save numpy array of tif image
print("successfully read image")

# 2) generate skeleton
# binArr = np.load(PATH + 'THR-' + IMG + '.npy')
resultSkel = thinVol.get_thinned(binArr)                # get skeleton
np.save(PATH + 'SKEL-' + IMG + '.npy', resultSkel)      # save skeleton as numpy array
tif.imsave(PATH + 'SKEL-' + IMG + '.tif', resultSkel.astype('int8'), bigtiff=True)  # save skeleton as tif image
print("successfully generated skeleton")

# 3) prune skeleton
# resultSkel = np.load(PATH + 'SKEL-' + IMG + '.npy')
# netGraphSkel = netGraphArr.get_networkx_graph_from_array(resultSkel)                # graph presentation of skeleton
# prunedSkel = getPrunedSkeleton(resultSkel, netGraphSkel)                            # get pruned skeleton
# np.save(PATH + 'PSKEL-' + IMG + '.npy', prunedSkel)                                 # save pruned skeleton as numpy array
# tif.imsave(PATH + 'PSKEL-' + IMG + '.tif', resultSkel.astype('int8'), bigtiff=True) # save pruned skeleton as tif image
# print("successfully generated pruned skeleton")

# 4) get statistics from generated skeletons
# resultSkel = np.load(PATH + 'SKEL-' + IMG + '.npy')
# prunedSkel = np.load(PATH + 'PSKEL-' + IMG + '.npy')
netGraphSkel = netGraphArr.get_networkx_graph_from_array(resultSkel)
# netGraphPSkel = netGraphArr.get_networkx_graph_from_array(prunedSkel)     # graph presentation of pruned skeleton

statsSkel = segStats.SegmentStats(netGraphSkel)
# statsPSkel = segStats.SegmentStats(netGraphPSkel)
statsSkel.setStats()
# statsPSkel.setStats()
print("successfully generated statistics")

# uncomment to show statistics
# print("---------Statistics----------")
# print("endPtsDict: ", statsSkel.countEndPointsDict)
# print("brPtsDict: ", statsSkel.countBranchPointsDict)
# print("sumLengthDict: ", statsSkel.sumLengthDict)
# print("lengthDict: ", statsSkel.lengthDict)
# print("contractionDict: ", statsSkel.contractionDict)
# print("angleDict: ", statsSkel.angleDict)

# save stats using pickle
pickle.dump(statsSkel, open(PATH + 'statsSKEL.p', 'wb'))
# pickle.dump(statsSkel, open(PATH + 'statsPSKEL.p', 'wb'))

# save statistics as CSV files in directory
statsDir = PATH + '/Statistics/'
os.makedirs(os.path.dirname(statsDir), exist_ok=True)

utils.saveFilamentDictAsCSV(statsSkel.countBranchPointsDict, statsDir + IMG + '_Filament_No._Segment_Branch_Points.csv',
                            'Filament No. Segment Branch Pts')
utils.saveFilamentDictAsCSV(statsSkel.countEndPointsDict, statsDir + IMG + '_Filament_No._Segment_Terminal_Points.csv',
                            'Filament No. Segment Terminal Pts')
utils.saveFilamentDictAsCSV(statsSkel.sumLengthDict, statsDir + IMG + '_Filament_Length_(sum).csv',
                            'Filament Length (sum)', 'um')
utils.saveSegmentDictAsCSV(statsSkel.lengthDict, statsDir + IMG + '_Segment_Length.csv', 'Segment Length', 'um')
utils.saveSegmentDictAsCSV(statsSkel.contractionDict, statsDir + IMG + '_Segment_Straightness.csv',
                           'Segment Straightness')
utils.saveSegmentDictAsCSV(statsSkel.angleDict, statsDir + IMG + '_Segment_Branching_Angle.csv',
                           'Segment Branching Angle', '°')

print("successfully saved statistics")

# 5) Visualization
#visualization = animation.getFrames(PATH + 'THR-' + IMG + '.npy', PATH + 'PSKEL-' + IMG + '.npy', 1)
