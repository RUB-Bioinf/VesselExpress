import csv
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ast import literal_eval as make_tuple
from collections import defaultdict
import imageio
import tifffile
import os


def saveAllStatsAsCSV(dictionary, path, imgName):
    # get all segment measurements as list from dictionary
    fil_id = 0
    key = 0
    for idx in dictionary:
        if bool(dictionary[idx]):
            key = next(iter(dictionary[idx]))
            fil_id = idx
            break
    ms_list = []
    for i in dictionary[fil_id][key].keys():
        ms_list.append(i)
    list = [["image", "filamentID", "segmentID"]]   # header list
    for i in ms_list:
        list[0].append(i)
    for filament in dictionary.keys():
        for segment in dictionary[filament]:
            list_item = [imgName, filament, segment]
            for stat in dictionary[filament][segment]:
                list_item.append(dictionary[filament][segment][stat])
            list.append(list_item)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(list)

def saveAllFilStatsAsCSV(dictionary, path, imgName):
    list = [["Image", "FilamentID", "No. Segments", "No. Terminal Points", "No. Branching Points"]]
    for filament in dictionary.keys():
        segs = dictionary[filament]["Segments"]
        endPts = dictionary[filament]["TerminalPoints"]
        brPts = dictionary[filament]["BranchPoints"]
        list_item = [imgName, filament, segs, endPts, brPts]
        list.append(list_item)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(list)

def saveBranchesBrPtAsCSV(dictionary, path, imgName):
    list = [["Image", "FilamentID", "BranchID", "No. Branches per BranchPoint"]]
    for filament in dictionary.keys():
        for segment in dictionary[filament]:
            branches = dictionary[filament][segment]
            list_item = [imgName, filament, segment, branches]
            list.append(list_item)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(list)

def saveEndPtsRelativeAsCSV(value, path, imgName):
    list = [["Image", "EndPts_ratio"]]
    list_item = [imgName, value]
    list.append(list_item)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(list)

def saveSegmentDictAsCSV(dictionary, path, measurementTitle, measurement, unit="", category="Segment"):
    """
        Save a dictionary with measurements as csv file

        Parameters
        ----------
        dictionary : defaultdict(dict)
        path : string of path to save csv of dict
        measurement : string of measurement
        unit : string of unit
        category : string of category
    """
    list = [[measurementTitle, "Unit", "Category", "FilamentID", category+"ID"]]
    for filament in dictionary.keys():
        for branch in dictionary[filament]:
            val = dictionary[filament][branch][measurement]
            list_item = [val, unit, category, filament, branch]
            list.append(list_item)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(list)

def saveBranchPtDictAsCSV(dictionary, path, measurementTitle, unit="", category="Branch"):
    """
        Save a dictionary with measurements as csv file

        Parameters
        ----------
        dictionary : defaultdict(dict)
        path : string of path to save csv of dict
        measurement : string of measurement
        unit : string of unit
        category : string of category
    """
    list = [[measurementTitle, "Unit", "Category", "FilamentID", "BranchID"]]
    for filament in dictionary.keys():
        for branch in dictionary[filament]:
            val = dictionary[filament][branch]
            list_item = [val, unit, category, filament, branch]
            list.append(list_item)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(list)

def saveFilamentDictAsCSV(dictionary, path, measurementTitle, measurement, unit=""):
    """
        Save a dictionary with measurements as csv file

        Parameters
        ----------
        dictionary : dict
        path : string of path to save csv of dict
        measurement : string of measurement
        unit : string of unit
    """
    list = [[measurementTitle, "Unit", "Category", "FilamentID"]]
    for filament in dictionary.keys():
        val = dictionary[filament][measurement]
        list_item = [val, unit, "Filament", filament]
        list.append(list_item)
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(list)

def plot3DGrid(arr, title):
    """
        Plots a 3D numpy array in a 3D grid with matplotlib

        Parameters
        ----------
        arr : 3D numpy array
        title : title of figure
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('z')
    ax.set_ylabel('y')
    ax.set_zlabel('x')

    ax.voxels(arr, edgecolor="k")
    plt.title(title)
    plt.show()

def plotSegStats(segmentsDict, brPtsDict, endPtsDict):
    """
        Plots every segment with its branch and end points in a 3D grid

        Parameters
        ----------
        segmentsDict : defaultdict(dict)
            A dictionary with the nth disjoint graph as the key containing a dictionary
            with key as the segment index (start node, end node) and value = list of nodes
        brPtsDict : dictionary with the nth disjoint graph as the key and the list of branch points as the value
        endPtsDict : dictionary with the nth disjoint graph as the key and the list of branch points as the value
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for filament in segmentsDict.keys():
        for branch in segmentsDict[filament]:
            # edges
            Z = [item[0] for item in segmentsDict[filament][branch]]
            Y = [item[1] for item in segmentsDict[filament][branch]]
            X = [item[2] for item in segmentsDict[filament][branch]]
            ax.plot(X, Y, Z, 'k')
            for point in segmentsDict[filament][branch]:
                # nodes
                if point in brPtsDict[filament]:
                    ax.scatter(point[2], point[1], point[0], c='r', marker='o')
                elif point in endPtsDict[filament]:
                    ax.scatter(point[2], point[1], point[0], c='g', marker='o')
                else:
                    ax.scatter(point[2], point[1], point[0], c='k', marker='o')

    ax.set_xlabel('z')
    ax.set_ylabel('y')
    ax.set_zlabel('x')
    plt.title('Graph with Branch and End Points')
    plt.show()

def getSegmentsDictFromFile(segmentsFile, fullSegDict, filamentNo):
    """
        Reads in segment keys of a filament from a file and creates a new dictionary containing the nodes of
        these segments as value

        Parameters
        ----------
        segmentsFile : file where each line contains the segment key as index (start node, end node)
        fullSegDict : defaultdict(dict)
            A dictionary with the nth disjoint graph as the key containing a dictionary
            with key as the segment index (start node, end node) and value = list of nodes
        filamentNo : no. of filament for the segments in fullSegDict

        Returns
        ----------
        segDict : defaultdict(dict)
            A dictionary with the filament no. as the key containing a dictionary
            with key as the segment index (start node, end node) and value = list of nodes
    """
    segDict = defaultdict(dict)
    file = open(segmentsFile, 'r')
    lines = file.readlines()

    for line in lines:
        segKey = line.strip()
        separator = segKey.index(')')
        first = make_tuple(segKey[1:separator+1])
        second = make_tuple(segKey[separator+3:-1])
        segKey = (first, second)
        if fullSegDict[filamentNo].get(segKey, False):
            segDict[filamentNo][segKey] = fullSegDict[filamentNo][segKey]

    return segDict

def read_img(filepath):
    extension = os.path.splitext(filepath)[1]
    if extension == '.tiff':
        img = tifffile.imread(filepath)
    else:
        img = imageio.imread(filepath)
    return img


def write_img(img, filepath):
    extension = os.path.splitext(filepath)[1]
    if extension == '.tiff':
        tifffile.imsave(filepath, img)
    else:
        imageio.imwrite(filepath, img)

