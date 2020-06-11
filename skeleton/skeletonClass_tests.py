from scipy import ndimage
import numpy as np

from skeleton.skeletonClass import Skeleton
from skeleton import skeleton_testlib


"""
Tests 3D thinning implemented as in
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999 is working as expected
from the program thinVolume.py
"""


def checkSameObjects(image):
    # check if number of objects are same in input and output of thinning
    label_img, countObjects = ndimage.measurements.label(image, structure=ndimage.generate_binary_structure(image.ndim, 2))
    skel = Skeleton(image)
    skel.setThinningOutput(mode="constant")
    thinned_blob = skel.skeletonStack
    label_img, countObjectsn = ndimage.measurements.label(thinned_blob, structure=ndimage.generate_binary_structure(image.ndim, 2))
    assert (countObjectsn == countObjects), ("number of objects in input "
                                             "{} is different from output {}".format(countObjects, countObjectsn))
    return thinned_blob


def checkAlgorithmPreservesImage(image):
    skel = Skeleton(image)
    skel.setThinningOutput(mode="constant")
    thinned_blob = skel.skeletonStack
    assert np.array_equal(image, thinned_blob)


def checkCycle(image):
    # check if number of cycles in the donut image after thinning is 1
    skel = Skeleton(image)
    skel.setThinningOutput(mode="constant")
    thinned_blob = skel.skeletonStack
    label_img, countObjects = ndimage.measurements.label(thinned_blob, structure=np.ones((3, 3, 3), dtype=bool))
    assert countObjects == 1, "number of cycles in single donut is {}".format(countObjects)


def test_donut():
    # Test 1 donut should result in a single cycle
    image = skeleton_testlib.get_donut()
    checkCycle(image)


def test_rectangles():
    # Test 3 All Rectangles should preserve topology and should have
    # same number of objects
    testImages = skeleton_testlib.getStationary3dRectangles()
    for image in testImages:
        print(image.sum())
        yield checkSameObjects, image


def test_singlePixelLines():
    # Test 4 single pixel lines should still be the same in an image
    checkSameObjects(skeleton_testlib.get_single_voxel_line())


def test_tinyLoopWithBranches():
    # Test 5 tiny loop with  branches should still be the same
    checkSameObjects(skeleton_testlib.get_tiny_loop_with_branches())


def test_wideLines():
    # Test 6 All widelines should preserve topology and should have
    # same number of objects
    testImages = skeleton_testlib.get3DRolledThickLines()
    for image in testImages:
        yield checkSameObjects, image


def test_crosPair():
    # Test 7 tiny loop with  branches should still be the same
    checkSameObjects(skeleton_testlib.get_disjoint_crosses())


def test_singleVoxelLine():
    # Test 8 single voxel line should still be the same
    checkSameObjects(skeleton_testlib.get_single_voxel_line())
