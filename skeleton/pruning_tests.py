import numpy as np
from scipy import ndimage

from skeleton.skeletonClass import Skeleton

"""
Program to test if pruned skeleton is as expected
"""


def getRectangle(w=10, h=6, d=6):
    # generate a rectangle
    return np.ones((w, h, d), dtype=bool)


def getRectangleNoise(w=10, h=6, d=6):
    # generate a rectangle with noise
    rect = np.zeros((w + 2, h + 2, d + 2), dtype=bool)
    rect[1: w + 1, 1: h + 1, 1: d + 1] = 1
    rect[int((w / 2) + 1), int((h / 2) + 1), 1] = 1
    return rect


def checkAlgorithmSameObjects(image):
    # check if same objects after and before pruning
    skel = Skeleton(image)
    skel.setPrunedSkeletonOutput()
    label_img, countObjectsn = ndimage.measurements.label(skel.outputStack, structure=np.ones((3, 3, 3), dtype=np.uint8))
    label_img, countObjects = ndimage.measurements.label(image, structure=np.ones((3, 3, 3), dtype=np.uint8))
    assert (countObjectsn == countObjects) or np.sum(label_img == 2) == 1


def test_rectangle():
    # Test 1 rectangle shouldn't do any pruning
    checkAlgorithmSameObjects(getRectangle())


def test_rectangleNoise():
    # Test 2 rectangle with noise must prune
    checkAlgorithmSameObjects(getRectangleNoise())
