import os
import numpy as np
# NOTE This does the pyx compilation of this extension
import pyximport; pyximport.install() # NOQA
import skeleton.thinVolume as thinVol
from skimage import io
import utils
import statistics.graph as graph
from skimage.morphology import skeletonize_3d
import time
import matlab.engine
#from memory_profiler import profile
import skeleton.networkx_graph_from_array as netGraphArr

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

    # read segmentation tif as numpy array
    begin = time.time()
    im = io.imread(imgPath + imgName + '.tif')              # reads image as array (z, y, x)
    binArr = im / np.max(im)                                # make binary array of 3D image/ thresholded 3D volume
    print("time taken to read segmentation tif is %0.3f seconds" % (time.time() - begin))

    # 2) generate skeleton
    if skeletonMethod != 'matlab':
        if skeletonMethod == 'scikit':
            start = time.time()
            resultSkel = skeletonize_3d(binArr)
            resultSkel = resultSkel / np.max(resultSkel)
            print("time taken to calculate skeleton is %0.3f seconds" % (time.time() - start))
        elif skeletonMethod == '3scan':
            resultSkel = thinVol.get_thinned(binArr)  # get skeleton
            resultSkel = resultSkel*1.0
        print("time taken for complete skeletonization is %0.3f seconds" % (time.time() - begin))
    else:
        begin = time.time()
        eng = matlab.engine.start_matlab()
        print("time taken to start matlab engine is %0.3f seconds" % (time.time() - begin))
        eng.cd(r'/Users/philippaspangenberg/Documents/ps-3scan-skeleton/skeleton3d-matlab-master', nargout=0)
        eng.matlabSkeletonize(imgPath, imgName, nargout=0)
        eng.quit()
        # retrieve numpy array from skeleton tif
        skel = io.imread(imgPath + 'skeleton-' + imgName + '.tif')  # reads image as array (z, y, x)
        resultSkel = skel / np.max(skel)
        print("time taken for complete skeletonization is %0.3f seconds" % (time.time() - begin))

    # 3) extract graph
    networkxGraph = netGraphArr.get_networkx_graph_from_array(resultSkel)

    # 4) calculate statistics from segmentation and skeleton mask
    statsSkel = graph.Graph(binArr, resultSkel, networkxGraph)
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
        utils.plot3DGrid(binArr, 'Segmentation Mask')  # plot 3D grid of binary image
        utils.plot3DGrid(resultSkel, 'Skeleton Mask')  # plot 3D grid of skeleton
        utils.plotSegStats(statsSkel.segmentsDict, statsSkel.branchPointsDict, statsSkel.endPointsDict)

executeVesselAnalysisPipeline('/Users/philippaspangenberg/Desktop/synth_tubes/tube_9/', 'seg_tube_9_intensity_variation_noise', 'scikit')