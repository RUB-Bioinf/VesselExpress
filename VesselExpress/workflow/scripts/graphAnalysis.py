import time
import numpy as np
import argparse
import networkx as nx
import sys
import os
from shutil import rmtree

# import modules
package = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'modules/'))
sys.path.append(package)

import networkx_graph_from_array as netGrArr
import graph
import utils


def processImage(skelImg, binImg, parameterDict):
    input_file = os.path.abspath(skelImg).replace('\\', '/')
    dir = os.path.dirname(input_file)
    file_name = os.path.basename(dir)

    # change to receive info file
    finfo = None
    #finfo = dir + '/' + file_name + '_info.csv'

    # Graph construction
    binImage = utils.read_img(binImg)
    binImage = binImage / np.max(binImage)
    skeleton = utils.read_img(skelImg)
    if np.max(skeleton) != 0:   # only if skeleton points are present
        skeleton = skeleton / np.max(skeleton)

    networkxGraph = netGrArr.get_networkx_graph_from_array(skeleton)

    # Statistical Analysis
    stats = graph.Graph(binImage, skeleton, networkxGraph, parameterDict.get("pixel_dimensions"),
                        pruningScale=parameterDict.get("pruning_scale"), lengthLimit=parameterDict.get("length_limit"),
                        diaScale=parameterDict.get("dia_scale"), branchingThreshold=parameterDict.get("branching_threshold"),
                        expFlag=parameterDict.get("experimental_flag"), infoFile=finfo,
                        graphCreation=parameterDict.get("extended_output"), smallRAMmode=parameterDict.get("small_RAM_mode"),
                        fileName=file_name, removeBorderEndPts=parameterDict.get("remove_border_end_pts"),
                        removeEndPtsFromSmallFilaments=parameterDict.get("remove_end_pts_from_small_filaments"),
                        interpolate=parameterDict.get("seg_interpolate"),
                        cut_neighbor_brpt_segs=parameterDict.get("cut_neighbor_brpt_segs"))
    stats.setStats()

    if parameterDict.get("extended_output") == 1:
        # save graph as image and graphml file
        graph_arr = stats.skeleton
        utils.write_img((graph_arr * 255).astype('uint8'), dir + '/Graph_' + file_name + '.'
                        + input_file.split('.')[1])
        g = stats.networkxGraph
        nx.write_graphml_lxml(g, dir + '/' + file_name + ".graphml")

        # save image with branch points
        brPts = []
        for i in stats.branchPointsDict.values():
            if i.keys():
                for k in i.keys():
                    brPts.append(k)
        brPt_img = np.zeros(graph_arr.shape)
        for ind in brPts:
            brPt_img[ind] = 255
        utils.write_img(brPt_img.astype('uint8'), dir + '/BrPts_' + file_name + '.'
                        + input_file.split('.')[1])

        # save image with terminal points
        endPts = []
        for i in stats.endPointsDict.values():
            for l in i:
                endPts.append(l)
        endPt_img = np.zeros(graph_arr.shape)
        for ind in endPts:
            endPt_img[ind] = 255
        utils.write_img(endPt_img.astype('uint8'), dir + '/EndPts_' + file_name + '.'
                        + input_file.split('.')[1])

    # Export statistics to csv files
    # os.makedirs(statsDir, exist_ok=True)
    #
    # utils.saveFilamentDictAsCSV(stats.filStatsDict, os.path.join(statsDir, file_name +
    #                             '_Filament_No._Segment_Branch_Points.csv'), 'Filament No. Segment Branch Pts',
    #                             'BranchPoints')
    # utils.saveFilamentDictAsCSV(stats.filStatsDict, os.path.join(statsDir, file_name +
    #                             '_Filament_No._Segment_Terminal_Points.csv'), 'Filament No. Segment Terminal Pts',
    #                             'TerminalPoints')
    # utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_Length.csv'),
    #                                 'Segment Length', 'length', 'um')
    # utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_Straightness.csv'),
    #                                 'Segment Straightness', 'straightness')
    # utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_Branching_Angle.csv'),
    #                                 'Segment Branching Angle', 'branchingAngle', '°')
    # utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_Volume.csv'),
    #                                 'Segment Volume', 'volume', 'um^3')
    # utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_Diameter.csv'),
    #                                 'Segment Diameter', 'diameter', 'um')
    # utils.saveBranchPtDictAsCSV(stats.branchPointsDict, os.path.join(statsDir, file_name + '_BranchPt_No._Branches.csv'),
    #                                  'BranchPt No. Branches', category='Branch')

    # create files containing all statisics in one csv per category (segment, filament, branches and endPtsRatio)
    utils.saveAllStatsAsCSV(stats.segStatsDict, dir + '.' + input_file.split('.')[1] + '_Segment_Statistics.csv', file_name)
    utils.saveAllFilStatsAsCSV(stats.filStatsDict, dir + '.' + input_file.split('.')[1] + '_Filament_Statistics.csv', file_name)
    utils.saveBranchesBrPtAsCSV(stats.branchesBrPtDict, dir + '.' + input_file.split('.')[1] + '_BranchesPerBranchPt.csv', file_name)
    if parameterDict.get("experimental_flag") == 1:
        statsDir = os.path.join(dir, 'statistics')
        os.makedirs(statsDir, exist_ok=True)
        utils.saveSegmentDictAsCSV(stats.segStatsDict, os.path.join(statsDir, file_name + '_Segment_z_Angle.csv'),
                                   'Segment z Angle', 'zAngle', '°')
        utils.saveEndPtsRelativeAsCSV(stats.endPtsTopVsBottom, dir + '_EndPtsRatio.csv', file_name)

    if parameterDict.get('small_RAM_mode'):
        rmtree('tmp_zarr' + os.sep + file_name + '_radiusMatrix.zarr')


if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes graph analysis on skeleton image file of type .tif')
    parser.add_argument('-skel_img', type=str, help='input skeleton image file to process')
    parser.add_argument('-bin_img', type=str, help='input binary image file to process')
    parser.add_argument('-pixel_dimensions', type=str, default="2.0,1.015625,1.015625",
                        help='Pixel dimensions in [z, y, x]')
    parser.add_argument('-pruning_scale', type=float, default=1.5,
                        help='Pruning scale for insignificant branch removal')
    parser.add_argument('-length_limit', type=float, default=3, help='Limit of vessel lengths')
    parser.add_argument('-dia_scale', type=float, default=2, help='Segments with lengths shorter than their diameter '
                                                                  'multiplied by this scaling factor are removed')
    parser.add_argument('-branching_threshold', type=float, default=0.25,
                        help='segments length as vector estimate for branching angle calculation')
    parser.add_argument('-extended_output', type=int, default=0,
                        help='if set to 1 outputs tif files with branch and terminal points and graph')
    parser.add_argument('-experimental_flag', type=int, default=0,
                        help='set to 1 for experimental statistics')
    parser.add_argument('-remove_border_end_pts', type=int, default=0, help='set to 1 to remove terminal points at the '
                                                                            'volume border')
    parser.add_argument('-remove_end_pts_from_small_filaments', type=int, default=0,
                        help='set to 1 to remove terminal points from small filaments with less than 5 segments')
    parser.add_argument('-seg_interpolate', type=int, default=0, help='set to 1 to interpolate segments')
    parser.add_argument('-cut_neighbor_brpt_segs', type=int, default=1, help='set to 0 to not cut segments which '
                                                                             'consists of 2 neighboring branch points')
    parser.add_argument('-small_RAM_mode', type=int, default=0, help='set to 1 for small RAM mode')
    parser.add_argument('-prints', type=bool, default=False, help='set to True to print runtime')
    args = parser.parse_args()

    pixel_dims = [float(item) for item in args.pixel_dimensions.split(',')]
    if pixel_dims[0] == 0:
        pixel_dims = pixel_dims[1:]

    parameters = {
        "pixel_dimensions": pixel_dims,
        "pruning_scale": args.pruning_scale,
        "length_limit": args.length_limit,
        "dia_scale": args.dia_scale,
        "branching_threshold": args.branching_threshold,
        "extended_output": args.extended_output,
        "experimental_flag": args.experimental_flag,
        "remove_border_end_pts": args.remove_border_end_pts,
        "remove_end_pts_from_small_filaments": args.remove_end_pts_from_small_filaments,
        "seg_interpolate": args.seg_interpolate,
        "cut_neighbor_brpt_segs": args.cut_neighbor_brpt_segs,
        "small_RAM_mode": args.small_RAM_mode
    }

    processImage(args.skel_img, args.bin_img, parameters)

    if args.prints:
        print("Graph extraction and statistical analysis completed in %0.3f seconds" % (time.time() - programStart))
