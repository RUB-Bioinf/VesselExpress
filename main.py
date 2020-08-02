import numpy as np
import os
# NOTE This does the pyx compilation of this extension
import pyximport; pyximport.install() # NOQA
import skeleton.thinVolume as thinVol
from skimage import io
from skimage.external import tifffile as tif
import utils
import statistics.graph as graph
from skimage.morphology import skeletonize_3d
import time
import argparse

def executeVesselAnalysisPipeline(imgPath, imgName, skeletonMethod='scikit', debug=False):
    """
        This is a pipeline containing the following steps for retrieving statistics from 3D vessel images:

        1. ToDo: segment vessel image
        2. generate skeleton
        3. retrieve network graph
        4. calculate statistics

        Parameters
        ----------
        imgPath : string of image path
        imgName : string of image name
        skeletonMethod : string of skeletonization method
        debug : bool
            if debug, then plots are generated for each step
    """

    # 1) read tif image as numpy array
    start = time.time()
    im = io.imread(imgPath + imgName + '.tif')              # reads image as array (z, y, x)
    binArr = im / np.max(im)                                # make binary array of 3D image/ thresholded 3D volume
    np.save(imgPath + 'THR-' + imgName + '.npy', binArr)    # save numpy array of tif image
    print("time taken to read and binarize image is %0.3f seconds" % (time.time() - start))

    if debug:
        utils.plot3DGrid(binArr, 'Segmentation Mask')   # plot 3D grid of binary image

    # 2) generate skeleton
    if skeletonMethod == 'scikit':
        start = time.time()
        resultSkel = skeletonize_3d(binArr)
        resultSkel = resultSkel / np.max(resultSkel)
        print("time taken to calculate skeleton is %0.3f seconds" % (time.time() - start))
    elif skeletonMethod == '3scan':
        resultSkel = thinVol.get_thinned(binArr)  # get skeleton
        resultSkel = resultSkel*1.0

    # save skeleton as numpy array and as tif image
    np.save(imgPath + 'SKEL-' + imgName + '.npy', resultSkel)
    tif.imsave(imgPath + 'SKEL-' + imgName + '.tif', resultSkel, bigtiff=True)

    if debug:
        utils.plot3DGrid(resultSkel, 'Skeleton Mask')    # plot 3D grid of skeleton

    # 3) calculate statistics from segmentation and skeleton mask
    statsSkel = graph.Graph(binArr, resultSkel)
    statsSkel.setStats()

    # save statistics as CSV files in directory
    statsDir = imgPath + '/Statistics_' + skeletonMethod + '/'
    os.makedirs(os.path.dirname(statsDir), exist_ok=True)

    utils.saveFilamentDictAsCSV(statsSkel.countBranchPointsDict, statsDir + imgName +
                                '_Filament_No._Segment_Branch_Points.csv', 'Filament No. Segment Branch Pts')
    utils.saveFilamentDictAsCSV(statsSkel.countEndPointsDict, statsDir + imgName +
                                '_Filament_No._Segment_Terminal_Points.csv', 'Filament No. Segment Terminal Pts')
    utils.saveFilamentDictAsCSV(statsSkel.sumLengthDict, statsDir + imgName + '_Filament_Length_(sum).csv',
                                'Filament Length (sum)', 'um')
    utils.saveSegmentDictAsCSV(statsSkel.lengthDict, statsDir + imgName + '_Segment_Length.csv', 'Segment Length', 'um')
    utils.saveSegmentDictAsCSV(statsSkel.straightnessDict, statsDir + imgName + '_Segment_Straightness.csv',
                               'Segment Straightness')
    utils.saveSegmentDictAsCSV(statsSkel.degreeDict, statsDir + imgName + '_Segment_Branching_Angle.csv',
                               'Segment Branching Angle', '°')
    utils.saveSegmentDictAsCSV(statsSkel.volumeDict, statsDir + imgName + '_Segment_Volume.csv',
                               'Segment Volume', 'um^3')
    utils.saveSegmentDictAsCSV(statsSkel.diameterDict, statsDir + imgName + '_Segment_Diameter.csv',
                               'Segment Diameter', 'um')
    utils.saveSegmentDictAsCSV(statsSkel.branchPointsDict, statsDir + imgName + '_BranchPt_No._Branches.csv',
                               'BranchPt No. Branches', category='Branch')
    print("successfully saved statistics")

    if debug:
        utils.plotSegStats(statsSkel.segmentsDict, statsSkel.branchPointsDict, statsSkel.endPointsDict)


if __name__ == '__main__':
    '''
        Instead of using the main, you can directly use the executeVesselAnalysisPipeline!

        example call:
            python main.py /Users/philippaspangenberg/Documents/TestBilder/B5_contra_RB_cmp/ C2-B5_contra_RB_cmp 
            scikit --debug False
    '''

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('imagePath', type=str, help='Path of image')
    parser.add_argument('imageName', type=str, help='Name of image')
    parser.add_argument('skeletonization', type=str, default='scikit', help='Which skeletonization method?')
    parser.add_argument('--debug', type=bool, default=False, action='store', dest='debug', help='Show plots?')
    results = parser.parse_args()

    executeVesselAnalysisPipeline(
        str(results.imagePath),         # path of image
        str(results.imageName),         # name of image
        str(results.skeletonization),   # skeletonization method
        debug=bool(results.debug)       # show plots?
    )

