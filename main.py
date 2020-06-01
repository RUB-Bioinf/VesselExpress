import numpy as np
# NOTE This does the pyx compilation of this extension
import pyximport; pyximport.install() # NOQA
import skeleton.thinVolume as thinVol
from skimage import io
from skimage.external import tifffile as tif
import skeleton.networkx_graph_from_array as netGraphArr
import metrics.segmentStats as segStats
from runscripts import animation
import utils
import os
import statistics.graph as graph

# specify path directory and image name
PATH = '/Users/philippaspangenberg/Documents/TestBilder/B6_contra_RB_cmp/'
IMG = 'C2-B6_contra_RB_cmp'

if __name__ == "__main__":

    # 1) read tif image as numpy array
    # im = io.imread(PATH + IMG + '.tif')                     # reads image as array (z, y, x)
    # binArr = im/np.max(im)                                  # make binary array of 3D image/ thresholded 3D volume
    # np.save(PATH + 'THR-' + IMG + '.npy', binArr)           # save numpy array of tif image
    # print("successfully read image")

    # 2) generate skeleton
    # binArr = np.load(PATH + 'THR-' + IMG + '.npy')
    # resultSkel = thinVol.get_thinned(binArr)                # get skeleton
    # np.save(PATH + 'SKEL-' + IMG + '.npy', resultSkel)      # save skeleton as numpy array
    # tif.imsave(PATH + 'SKEL-' + IMG + '.tif', resultSkel.astype('int8'), bigtiff=True)  # save skeleton as tif image
    # print("successfully generated skeleton")

    # 3) get statistics from generated skeletons
    resultSkel = np.load(PATH + 'SKEL-' + IMG + '.npy')
    netGraphSkel = netGraphArr.get_networkx_graph_from_array(resultSkel)

    # statsSkel = segStats.SegmentStats(netGraphSkel)
    # statsSkel.setStats()
    statsSkel = graph.Graph(netGraphSkel)
    statsSkel.setStats()
    print("successfully generated statistics")

    # uncomment to show statistics
    # print("---------Statistics----------")
    # print("endPtsDict: ", statsSkel.countEndPointsDict)
    # print("brPtsDict: ", statsSkel.countBranchPointsDict)
    # print("sumLengthDict: ", statsSkel.sumLengthDict)
    # print("lengthDict: ", statsSkel.lengthDict)
    # print("straightnessDict: ", statsSkel.straightnessDict)
    # print("angleDict: ", statsSkel.degreeDict)

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
    utils.saveSegmentDictAsCSV(statsSkel.straightnessDict, statsDir + IMG + '_Segment_Straightness.csv',
                               'Segment Straightness')
    utils.saveSegmentDictAsCSV(statsSkel.degreeDict, statsDir + IMG + '_Segment_Branching_Angle.csv',
                               'Segment Branching Angle', '°')
    print("successfully saved statistics")

    # 4) Visualization
    #visualization = animation.getFrames(PATH + 'THR-' + IMG + '.npy', PATH + 'PSKEL-' + IMG + '.npy', 1)
