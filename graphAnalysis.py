import time
import numpy as np
import os
import argparse

from Graph import networkx_graph_from_array as netGrArr
import Statistics.graph as graph
import Statistics.utils as stat_utils
import utils


def processImage(skelImg, binImg, statsDir, parameterDict):
    input_file = os.path.abspath(skelImg).replace('\\', '/')
    dir = os.path.dirname(input_file)
    file_name = os.path.basename(dir)

    # change to receive info file
    finfo = None
    #finfo = dir + '/' + file_name + '_info.csv'

    # Graph construction
    print("graph construction...")
    binImage = utils.read_img(binImg)
    binImage = binImage / np.max(binImage)
    skeleton = utils.read_img(skelImg)
    if np.max(skeleton) != 0:   # only if skeleton points are present
        skeleton = skeleton / np.max(skeleton)

    networkxGraph = netGrArr.get_networkx_graph_from_array(skeleton)

    # Statistical Analysis
    stats = graph.Graph(binImage, skeleton, networkxGraph, parameterDict.get("pixel_dimensions"),
                        pruningScale=parameterDict.get("pruning_scale"), lengthLimit=parameterDict.get("length_limit"),
                        branchingThreshold=parameterDict.get("branching_threshold"),
                        expFlag=parameterDict.get("experimental_flag"), infoFile=finfo)
    stats.setStats()

    # Export statistics to csv files
    os.makedirs(statsDir, exist_ok=True)

    stat_utils.saveFilamentDictAsCSV(stats.filStatsDict, os.path.join(statsDir, file_name +
                                '_Filament_No._Segment_Branch_Points.csv'), 'Filament No. Segment Branch Pts',
                                'BranchPoints')
    stat_utils.saveFilamentDictAsCSV(stats.filStatsDict, os.path.join(statsDir, file_name +
                                '_Filament_No._Segment_Terminal_Points.csv'), 'Filament No. Segment Terminal Pts',
                                'TerminalPoints')
    stat_utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_Length.csv'),
                                    'Segment Length', 'length', 'um')
    stat_utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_Straightness.csv'),
                                    'Segment Straightness', 'straightness')
    stat_utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_Branching_Angle.csv'),
                                    'Segment Branching Angle', 'branchingAngle', '°')
    stat_utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_Volume.csv'),
                                    'Segment Volume', 'volume', 'um^3')
    stat_utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_Diameter.csv'),
                                    'Segment Diameter', 'diameter', 'um')
    stat_utils.saveBranchPtDictAsCSV(stats.branchPointsDict, os.path.join(statsDir, file_name + '_BranchPt_No._Branches.csv'),
                                     'BranchPt No. Branches', category='Branch')
    if parameterDict.get("experimental_flag") == 1:
        stat_utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_z_Angle.csv'),
                                        'Segment z Angle', 'zAngle', '°')

    # create files containing all statisics in one csv per category (segment, filament, branches and endPtsRatio)
    if parameterDict.get("all_stats") == 1:
        stat_utils.saveAllStatsAsCSV(stats.segStatsDict, dir + '_All_Segment_Statistics.csv', file_name)
        stat_utils.saveAllFilStatsAsCSV(stats.filStatsDict, dir + '_All_Filament_Statistics.csv', file_name)
        stat_utils.saveBranchesBrPtAsCSV(stats.branchesBrPtDict, dir + '_All_BranchesPerBranchPt.csv', file_name)
        if parameterDict.get("experimental_flag") == 1:
            stat_utils.saveEndPtsRelativeAsCSV(stats.endPtsTopVsBottom, dir + '_EndPtsRatio.csv', file_name)


if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes graph analysis on skeleton image file of type .tif')
    parser.add_argument('-skel_img', type=str, help='input skeleton image file to process')
    parser.add_argument('-bin_img', type=str, help='input binary image file to process')
    parser.add_argument('-output_dir', type=str, help='directory to store statistics')
    parser.add_argument('-pixel_dimensions', type=str, default="2.0,1.015625,1.015625",
                        help='Pixel dimensions in [z, y, x]')
    parser.add_argument('-pruning_scale', type=float, default=1.5,
                        help='Pruning scale for insignificant branch removal')
    parser.add_argument('-length_limit', type=float, default=3, help='Limit of vessel lengths')
    parser.add_argument('-branching_threshold', type=float, default=0.25,
                        help='segments length as vector estimate for branching angle calculation')
    parser.add_argument('-all_stats', type=int, default=0,
                        help='if set to 1 create CSVs containing all statistics for each category')
    parser.add_argument('-experimental_flag', type=int, default=0,
                        help='set to 1 for experimental statistics')
    args = parser.parse_args()

    parameters = {
        "pixel_dimensions": [float(item) for item in args.pixel_dimensions.split(',')],
        "pruning_scale": args.pruning_scale,
        "length_limit": args.length_limit,
        "branching_threshold": args.branching_threshold,
        "all_stats": args.all_stats,
        "experimental_flag": args.experimental_flag
    }

    processImage(args.skel_img, args.bin_img, args.output_dir, parameters)

    print("Graph extraction and statistical analysis completed in %0.3f seconds" % (time.time() - programStart))
