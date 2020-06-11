import itertools

import nose.tools
import numpy as np
import scipy.ndimage as ndimage

import skeleton.skeleton_testlib as skeleton_testlib
import skeleton.thinVolume as thin_volume
import skeleton.image_tools as image_tools

"""
Tests for 3D thinning algorithms testing strictly with change in directions, axis
3D Thinning implemented using
A Parallel 3D 12-Subiteration Thinning Algorithm
Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
"""


def embed_2d_in_3d(arr):
    # Embed a 2D shape in a 3D array, along all possible testing directions.

    m, n = arr.shape
    embedded_in_x = np.zeros((3, m, n), dtype=bool)
    embedded_in_x[1, :, :] = arr

    embedded_in_y = np.zeros((m, 3, n), dtype=bool)
    embedded_in_y[:, 1, :] = arr

    embedded_in_z = np.zeros((m, n, 3), dtype=bool)
    embedded_in_z[:, :, 1] = arr

    return embedded_in_x, embedded_in_y, embedded_in_z


def reorders(arr):
    # reorder the slices, rows, columns of 3D stack
    for xf, yf, zf in itertools.combinations_with_replacement([1, -1], 3):
        yield arr[::xf, ::yf, ::zf]


def do_embedded_helper(arr, expected_result=None):
    # a 2D slice is embedded as different plane (XY, YZ, XZ) in a 3D volume
    # and tested if number of objects are as expected after thinning
    two_result = _get_count_objects(arr)

    if expected_result is not None:
        nose.tools.assert_equal(two_result, expected_result, msg=arr.shape)
    else:
        expected_result = two_result

    for embedding in embed_2d_in_3d(arr):
        _all_orientations_helper(embedding, expected_result)
    return two_result


def _all_orientations_helper(arr, expected_result=None):
    # different 90 degree rotations of a 3D volume and test if number of objects are as
    # expected after thinning
    for reoriented in reorders(arr):
        thinned_volume = thin_volume.get_thinned(reoriented, mode='constant')
        result = _get_count_objects(thinned_volume)
        nose.tools.assert_equal(result, expected_result)


def _get_count_objects(image):
    # count number of 26 or 8 connected objects in the skeletonized image
    label_skel, countObjects = ndimage.measurements.label(
        image, ndimage.generate_binary_structure(image.ndim, 2))
    return countObjects


def get_tiny_loop():
    # One single tiny loop
    tiny_loop = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=bool)
    return tiny_loop


def get_frame():
    # Frame
    frame = np.zeros((20, 20), dtype=bool)
    frame[2:-2, 2:-2] = 1
    frame[4:-4, 4:-4] = 0
    return frame


def get_ring(ri, ro, size=(25, 25)):
    # Make a annular ring in 2d. The inner and outer radius are given as a
    # percentage of the overall size.
    n, m = size
    xs, ys = np.mgrid[-1:1:n * 1j, -1:1:m * 1j]
    r = np.sqrt(xs ** 2 + ys ** 2)

    torus = np.zeros(size, dtype=bool)
    torus[(r < ro) & (r > ri)] = 1
    return torus


def test_simple_loop_embedded():
    # Test 1 a single loop embedded in different axis locations
    do_embedded_helper(get_tiny_loop(), 1)


def test_multi_loop_embedded():
    # Test 2 Three independent loops embedded in different axis locations
    tiny_loop = get_tiny_loop()
    multi_loop = np.zeros((25, 25), dtype=bool)
    multi_loop[2:5, 2:5] = tiny_loop
    multi_loop[7:10, 7:10] = tiny_loop
    do_embedded_helper(multi_loop, 2)


def test_cross_embedded():
    # Test 3 cross embedded in different axis locations
    cross = np.zeros((25, 25), dtype=bool)
    cross[:, 12] = 1
    cross[12, :] = 1
    do_embedded_helper(cross, 1)


def test_loop():
    # Test 4 Two joint loops embedded in different axis locations
    loop_pair = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=bool)
    do_embedded_helper(loop_pair, 1)


def test_square():
    # Test 5 Square embedded in different axis locations
    squae = np.zeros((20, 20), dtype=bool)
    squae[2:-2, 2:-2] = 1
    do_embedded_helper(squae, 1)


def test_frame():
    # Test 6 Frame (hollow square) embedded in different axis locations
    num_objects = do_embedded_helper(get_frame())
    nose.tools.assert_equal(num_objects, 1)


def test_framed_square():
    # Test 7 Square inside a Frame (hollow square) embedded in different axis locations
    framed_square = get_frame().copy()
    framed_square[6:-6, 6:-6] = 1
    num_objects = do_embedded_helper(framed_square)
    nose.tools.assert_equal(num_objects, 2)


def test_circle():
    # Test 8 Circle embedded in different axis locations
    circle = np.zeros((25, 25), dtype=bool)
    xs, ys = np.mgrid[-1:1:25j, -1:1:25j]

    for trial in range(5):
        circle[:] = 0
        r = np.random.uniform(3, 10)
        xc, yc = np.random.uniform(-1, 1, size=2)
        mask = ((xs ** 2) + (ys ** 2)) < r ** 2
        circle[mask] = 1

        num_objects = do_embedded_helper(circle)
        nose.tools.assert_equal(num_objects, 1)


def test_heaviside():
    # Test 9 Heaviside(comb) embedded in different axis locations
    heavi = np.zeros((20, 20), dtype=bool)
    heavi[10:, :] = 1
    do_embedded_helper(heavi, 1)


def test_ellipse():
    # Test 10 Ellipse embedded in different axis locations
    ellipse = np.zeros((25, 25), dtype=bool)
    xs, ys = np.mgrid[-1:1:25j, -1:1:25j]

    aspect = np.random.randint(1, 2) / 10

    for trial in range(5):
        ellipse[:] = 0
        r = np.random.uniform(3, 10)
        mask = (aspect * ((xs ** 2) + (ys ** 2))) < r ** 2
        ellipse[mask] = 1
        ellipse = ellipse.astype(bool)
        num_objects = do_embedded_helper(ellipse)
        nose.tools.assert_equal(num_objects, 1)


def test_concentric():
    # Test 11 Concentric circles embedded in different axis locations
    concentric_circles = get_ring(0.1, 0.2) + get_ring(0.4, 0.5) + get_ring(0.7, 0.9)
    num_circles = do_embedded_helper(concentric_circles)
    nose.tools.assert_equal(num_circles, 3)


def test_banana():
    # Test 12 Banana embedded in different axis locations
    # https://en.wikipedia.org/wiki/Rosenbrock_function
    xf, yf = np.mgrid[-1.5:3:50j, -1.5:2:50j]
    f = (1 - xf) ** 2 + 100 * (yf - xf ** 2) ** 2
    banana = 1 * (f > 250)
    banana = banana.astype(bool)
    do_embedded_helper(banana)


def test_hilbert_curve():
    # Test 13 Hilbert flipped in different orientations
    hilbert_curve = skeleton_testlib.get_hilbert_curve()
    _all_orientations_helper(hilbert_curve, expected_result=1)


def test_parallelepiped():
    # Test 14 Parallelepiped flipped in different orientations
    parallelepiped = np.zeros((10, 10, 10), dtype=bool)
    parallelepiped[2:-2, 2:-2, 2:-2] = 1
    _all_orientations_helper(parallelepiped, expected_result=1)


def test_frame3d():
    # Test 15 3D frame flipped in different orientations
    frame_3d = np.zeros((10, 10, 10), dtype=bool)
    frame_3d[2:-2, 2:-2, 2:-2] = 1
    frame_3d[4:-4, 4:-4, 4:-4] = 0
    _all_orientations_helper(frame_3d, 1)


def get_stationary_3d_rectangles(width=5):
    # Test #1:
    # the answer that skeletonize gives for h_line/v_line should be the same as
    # the input, as there is no pixel that can be removed without affecting
    # the topology.
    # The algorithim when run on any of the below feature sets
    # Should return a skeleton that is the same as the feature
    # A single horizontal/vertical line
    h_line = np.zeros((25, 25, 25), dtype=bool)
    h_line[:, 8:8 + width, :] = 1
    v_line = h_line.T.copy()

    # A "comb" of lines
    h_lines = np.zeros((25, 25, 25), dtype=bool)
    h_lines[0:width, ::3, :] = 1
    v_lines = h_lines.T.copy()
    # A grid made up of two perpendicular combs
    grid = h_lines | v_lines
    stationary_images = [h_line, v_line, h_lines, v_lines, grid]
    return stationary_images


def get_rand_images(width=4):
    # Test #2:
    # The algorithim should be able to run on arbitrary input without crashing.
    # We are not interested in the answer, so much as that the algo has full
    # coverage over the possible inputs into it
    random_images = [np.random.randint(2, size=(25, 25, 25), dtype=bool) for i in range(6)]
    return random_images


def get_thick_lines():
    # Test #3:
    # The algorithim should thing a single long contiguous segment to a line of
    # pixels
    bar_images = []
    # 2,6 and 20 pixel wide lines
    for i in [2, 6, 20]:
        h_line = np.zeros((25, 25, 25), dtype=bool)
        h_line[1, 1:i + 1, :] = 1
        v_line = h_line.T.copy()
        bar_images.append((h_line, i, (i + 1) // 2))
        bar_images.append((v_line, i, (i + 1) // 2))
    # Result graph should have _no_ cycles
    return bar_images


def test_random_images():
    for image in get_rand_images():
        thin_volume.get_thinned(image, mode='constant')


def test_rectangles():
    test_images = get_stationary_3d_rectangles(width=0)
    for image in test_images:
        np.testing.assert_array_equal(image, thin_volume.get_thinned(image, mode='constant'))


def test_wide_lines():
    test_images = get_thick_lines()
    for image, thickness, center in test_images:
        result = thin_volume.get_thinned(image, mode='reflect')
        nose.tools.assert_equal(result.sum(), image.sum() // thickness)
        nzc = image_tools.list_of_tuples_of_val(result, 1)
        prev_slope = np.subtract(nzc[0], nzc[1])
        for p1, p2 in zip(nzc[:-1], nzc[1:]):
            slope = np.subtract(p1, p2)
            # all points lie on same straight line
            np.testing.assert_array_equal(slope, prev_slope)
            # all points are in the center of the thick line
            nose.tools.assert_in(center, p1)
            nose.tools.assert_in(center, p2)
