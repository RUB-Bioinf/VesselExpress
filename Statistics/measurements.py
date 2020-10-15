import numpy as np
import math

def getLength(path, dimensions):
    """
        Find length of a path as distance between nodes in it

        Parameters
        ----------
        path : list
            list of nodes in the path

        dimensions : list
            list with pixel dimensions in desired unit (e.g. microns)
            3D: [z, y, x]   2D: [y, x]

        Returns
        -------
        length : float
            Length of path
    """
    length = 0
    for index, item in enumerate(path):
        if index + 1 != len(path):
            item2 = path[index + 1]
        vect = [j - i for i, j in zip(item, item2)]
        vect = [a * b for a, b in zip(vect, dimensions)]  # multiply pixel length with original length
        length += np.linalg.norm(vect)
    return length

def getVolume(skelRadii, segment):
    """
        Calculate volume and average diameter of a segment

        Parameters
        ----------
        segment : list
            list of nodes in the segment

        Returns
        -------
        volume, average diameter : float
    """
    volume = 0
    diameter = 0
    for skelPt in segment:
        volume = volume + math.pi * skelRadii[skelPt]**2
        diameter = diameter + skelRadii[skelPt] * 2
    return volume, diameter / len(segment)
