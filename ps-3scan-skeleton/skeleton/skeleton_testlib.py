import numpy as np
from scipy.spatial import ConvexHull

from skeleton.skeletonClass import Skeleton


def get_hilbert_curve():
    return np.array([[[1, 1, 1],
                      [1, 0, 1],
                      [1, 0, 1]],
                     [[0, 0, 0],
                      [0, 0, 0],
                      [1, 0, 1]],
                     [[1, 1, 1],
                      [1, 0, 1],
                      [1, 0, 1]]], dtype=bool)


def get_thinnedRandomBlob():
    # get random convex blob
    xs = np.random.uniform(-1, 1, size=50)
    ys = np.random.uniform(-1, 1, size=50)
    zs = np.random.uniform(-1, 1, size=50)

    xyzs = list(zip(xs, ys, zs))

    hullz = ConvexHull(xyzs)

    xf, yf, zf = np.mgrid[-1:1:100j, -1:1:10j, -1:1:10j]
    blob = np.ones(xf.shape, dtype=bool)
    for x, y, z, c in hullz.equations:
        mask = (xf * x) + (yf * y) + (zf * z) - c < 0
        blob[mask] = 0
    blob = blob.astype(bool)
    skel = Skeleton(blob)
    skel.setThinningOutput()
    thinned_blob = skel.skeletonStack
    return thinned_blob


def getStationary3dRectangles(width=5):
    # cubes of different sizes
    h_line = np.zeros((25, 25, 25), dtype=bool)
    h_line[:, 8:8 + width, :] = 1
    v_lines = h_line.T.copy()

    # A "comb" of lines
    h_lines = np.zeros((25, 25, 25), dtype=bool)
    h_lines[0:width, ::3, :] = 1
    v_liness = h_lines.T.copy()
    # A grid made up of two perpendicular combs
    grid = h_lines | v_liness
    stationaryImages = [h_line, v_lines, h_lines, v_liness, grid]
    return stationaryImages


def get3DRolledThickLines():
    # grid of thick lines
    hBar = np.zeros((25, 25, 25), dtype=bool)
    hBar[1, 0:5, :] = 1
    barImages = [np.roll(hBar, 2 * n, axis=0) for n in range(10)]
    return barImages


def getRing(ri, ro, size=(25, 25)):
    # Make a annular ring in 2d. The inner and outer radius are given as a
    # percentage of the overall size.
    n, m = size
    xs, ys = np.mgrid[-1:1:n * 1j, -1:1:m * 1j]
    r = np.sqrt(xs ** 2 + ys ** 2)

    torus = np.zeros(size, dtype=bool)
    torus[(r < ro) & (r > ri)] = 1
    return torus


def get_donut(width=2, size=(25, 25, 25)):
    # Ring of width = Donut
    x, y, z = size
    assert width < z / 2, "width {} of the donut should be less than half the array size in z {}".format(width, z / 2)

    # This is a single planr slice of ring
    ringPlane = getRing(0.25, 0.5, size=(x, y))

    # Stack up those slices starting form the center
    donutArray = np.zeros(size, dtype=bool)
    zStart = z // 2
    for n in range(width):
        donutArray[zStart + n, :, :] = ringPlane

    return donutArray


def get_tiny_loop_with_branches(size=(10, 10)):
    from skimage.morphology import skeletonize as getSkeletonize2D
    # a loop and a branches coming at end of the cycle
    frame = np.zeros(size, dtype=np.uint8)
    frame[2:-2, 2:-2] = 1
    frame[4:-4, 4:-4] = 0
    frame = getSkeletonize2D(frame)
    frame[1, 5] = 1
    frame[7, 5] = 1
    sampleImage = np.zeros((3, 10, 10), dtype=np.uint8)
    sampleImage[1] = frame
    return sampleImage


def get_disjoint_crosses(size=(10, 10, 10)):
    # two disjoint crosses
    crosPair = np.zeros(size, dtype=np.uint8)
    cros = np.zeros((5, 5), dtype=np.uint8)
    cros[:, 2] = 1
    cros[2, :] = 1
    crosPair[0, 0:5, 0:5] = cros
    crosPair[5, 5:10, 5:10] = cros
    return crosPair


def get_single_voxel_line(size=(5, 5, 5)):
    sampleLine = np.zeros(size, dtype=np.uint8)
    sampleLine[1, :, 4] = 1
    return sampleLine


def get_cycle_no_tree():
    # graph of a cycle
    donut = get_donut()
    skel = Skeleton(donut)
    skel.setNetworkGraph(True)
    return skel.graph


def get_cycles_with_branches_protrude():
    # graph of a cycle with branches
    sampleImage = get_tiny_loop_with_branches()
    skel = Skeleton(sampleImage)
    skel.setNetworkGraph(False)
    return skel.graph


def get_disjoint_trees_no_cycle_3d():
    # graph of two disjoint trees
    crosPair = get_disjoint_crosses()
    skel = Skeleton(crosPair)
    skel.setNetworkGraph(False)
    return skel.graph


def get_single_voxel_lineNobranches():
    # graph of no branches single line
    sampleLine = get_single_voxel_line()
    skel = Skeleton(sampleLine)
    skel.setNetworkGraph(False)
    return skel.graph
