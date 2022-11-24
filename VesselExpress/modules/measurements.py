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


def getRadius(distTrans, segment, smallRAMmode=0):
    sumRadii = 0
    if not smallRAMmode:
        for skelPt in segment:
            sumRadii += distTrans[skelPt]
    else:
        for i, skelPt in enumerate(segment):
            sumRadii += distTrans[skelPt]
            if i % 100 == 99:
                sumRadii = sumRadii.compute()
        sumRadii = sumRadii.compute()
    return sumRadii / len(segment)


def getVolumeCylinder(radius, segLength):
    return math.pi * radius ** 2 * segLength


def getVolume(skelRadii, segment, dimensions):
    volume = 0
    for index, skelPt in enumerate(segment):
        if index + 1 != len(segment):
            vect = [j - i for i, j in zip(skelPt, segment[index+1])]
            vect = [a * b for a, b in zip(vect, dimensions)]  # multiply with pixel dimensions
            rad = skelRadii[int(skelPt[0]), int(skelPt[1]), int(skelPt[2])]
            volume += math.pi * rad ** 2 * np.linalg.norm(vect)
    return volume

# def getVolume(skelRadii, segment, segLength, dimensions, fast=True):
#     """
#         Calculate volume and average diameter of a segment
#
#         Parameters
#         ----------
#         skelRadii: numpy array
#             array containing the distance to the closest background point for each voxel
#         segment : list
#             list of nodes in the segment
#         segLength: float
#             segment length
#         dimensions: list
#             pixel dimensions [z, y, x]
#         fast: bool
#             if true computes fast volume calculation
#
#         Returns
#         -------
#         volume, diameter : float
#     """
#     sumRadii = 0
#     # faster version takes the average radius of the whole segment and calculates it's volume
#     if fast:
#         for skelPt in segment:
#             sumRadii += skelRadii[skelPt]
#         avgRadius = sumRadii / len(segment)
#         diameter = avgRadius * 2
#         volume = math.pi * avgRadius ** 2 * segLength
#     # slower version calculates the surface for each pixel multiplied by the distance to the next pixel
#     else:
#         volume = 0
#         for index, skelPt in enumerate(segment):
#             if index + 1 != len(segment):
#                 vect = [j - i for i, j in zip(skelPt, segment[index+1])]
#                 vect = [a * b for a, b in zip(vect, dimensions)]  # multiply with pixel dimensions
#                 volume += math.pi * skelRadii[skelPt] ** 2 * np.linalg.norm(vect)
#             sumRadii += skelRadii[skelPt]
#         diameter = (sumRadii / len(segment)) * 2
#
#     return volume, diameter

def get_z_angle(segment, pixelDims):
    zVector = [1, 0, 0]
    v1 = segment[0]

    if segment[len(segment) - 1] == segment[0]:  # in case of a circle, take pre-last point
        v2 = segment[len(segment) - 2]
    else:
        v2 = segment[len(segment) - 1]

    dist_v1_z = np.linalg.norm(v1[1:])
    dist_v2_z = np.linalg.norm(v2[1:])

    if dist_v1_z < dist_v2_z:
        segVector = [j - i for i, j in zip(v1, v2)]  # v2-v1
    else:
        segVector = [j - i for i, j in zip(v2, v1)]  # v1-v2

    segVector = [a * b for a, b in zip(segVector, pixelDims)]

    cosine_angle = np.dot(zVector, segVector) / (np.linalg.norm(zVector) * np.linalg.norm(segVector))
    angle = np.arccos(round(cosine_angle, 4))

    return round(np.degrees(angle), 4)


