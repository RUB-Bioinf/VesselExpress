import nose.tools
import numpy as np
import scipy.ndimage.filters as sci_filter

# NOTE This does the pyx compilation of this extension
import pyximport; pyximport.install() # NOQA

import skeleton.rotational_operators as rop
import skeleton.thinning as thinning


def test_cy_convolve_basic():
    small_domain = np.zeros((5, 6, 7), dtype=np.uint64)
    kernel = np.zeros((3, 3, 3), dtype=np.uint64)
    kernel = kernel[::-1, ::-1, ::-1]
    points = np.zeros((2, 3), dtype=np.int)

    result = thinning.cy_convolve(small_domain, kernel, points, 'constant', 0)

    nose.tools.assert_equal(len(result), 2)
    np.testing.assert_array_equal(result, 0)


def test_cy_convolve():
    small_domain = np.ones((5, 6, 7), dtype=np.uint64)
    kernel = np.ones((3, 3, 3), dtype=np.uint64)
    kernel = kernel[::-1, ::-1, ::-1]
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int)

    result = thinning.cy_convolve(small_domain, kernel, points, 'constant', 0)

    nose.tools.assert_equal(len(result), 2)
    np.testing.assert_array_equal(result[0], 8)
    np.testing.assert_array_equal(result[1], 27)
    kernel = rop.DIRECTIONS_LIST[0]
    result = thinning.cy_convolve(small_domain, kernel, points, 'reflect', 0)
    nose.tools.assert_equal(result[0], 2 ** 26 - 1)
    nose.tools.assert_equal(result[1], 2 ** 26 - 1)


def test_cy_convolve_bounds():
    small_domain = np.ones((5, 6, 7), dtype=np.uint64)
    kernel = np.ones((3, 3, 3), dtype=np.uint64)
    kernel = kernel[::-1, ::-1, ::-1]
    points = np.array([[0, 0, 0]], dtype=np.int)

    result = thinning.cy_convolve(small_domain, kernel, points, 'reflect', 0)

    nose.tools.assert_equal(len(result), 1)
    np.testing.assert_array_equal(result, 27)


def test_cy_convole_vs_scipy():
    small_domain = np.random.randint(10, size=(5, 6, 7), dtype=np.uint64)
    kernel = np.random.randint(2, size=(3, 3, 3), dtype=np.uint64)
    kernel_flipped = kernel[::-1, ::-1, ::-1]
    np_result = sci_filter.convolve(small_domain, kernel)
    points = np.array([[1, 2, 3], [0, 0, 0]], dtype=np.int)

    result = thinning.cy_convolve(small_domain, kernel_flipped, points, 'reflect', 0)

    nose.tools.assert_equal(result[0], np_result[1, 2, 3])
    nose.tools.assert_equal(result[1], np_result[0, 0, 0])


# def test_cy_convolve_harder():
#     cylinder_radius = 5
#     shape = (32, 32, 32)
#     axis = 0
#     cylinder_in_axis = vessel_phantom.cylinder_on_axis(
#         radius=cylinder_radius, axis=axis, shape=shape)
#     cylinder_in_axis = (cylinder_in_axis / 255).astype(np.uint64)
#     points = np.array([[31, 12, 15]], dtype=np.int)
#     kernel = np.asarray(rop.DIRECTIONS_LIST[0])[tuple([slice(None, None, -1)] * 3)]
#     cylinder_skeleton = thinning.cy_convolve(cylinder_in_axis, kernel, points, 'reflect', 0)
#     # after convolving the reference array
#     # sum should be less than 2 ** 26 -1 (if all the 1st ordered neighborhood is 1)
#     nose.tools.assert_less_equal(cylinder_skeleton.sum(), 2 ** 26 - 1)


# def get_arr_from_coords(coords, shape):
#     border_arr = np.zeros(shape, dtype=bool)
#     for i, index in enumerate(coords):
#         border_arr[tuple(index)] = 1
#     return border_arr


# def test_get_border_coords():
#     cylinder_radius = 5
#     shape = (32, 32, 32)
#     for axis in range(3):
#         cylinder_in_axis = vessel_phantom.cylinder_on_axis(
#             radius=cylinder_radius, axis=axis, shape=shape)
#         cylinder_in_axis = cylinder_in_axis / 255
#         arr = cylinder_in_axis.astype(np.uint64)
#         non_zero_coords = np.asarray(thinning.kmath.list_of_tuples_of_val(
#             np.asarray(arr), 1), order='C')
#         border_coords = thinning.get_border_coords(arr, non_zero_coords, 'reflect', 0)
#         border_arr = get_arr_from_coords(border_coords, shape)
#         # Number of 6 connected objects should be 12
#         nose.tools.assert_equal(ndi.measurements.label(border_arr)[1], 12)


# def test_cy_get_thinned_cylinder():
#     cylinder_radius = 5
#     shape = (32, 32, 32)
#     for axis in range(3):
#         cylinder_in_axis = vessel_phantom.cylinder_on_axis(
#             radius=cylinder_radius, axis=axis, shape=shape)
#         cylinder_in_axis = cylinder_in_axis / 255
#         cylinder_skeleton = thinning.cy_get_thinned_3d(
#             cylinder_in_axis.astype(np.uint64), 'reflect', 0)
#         nose.tools.assert_equal(cylinder_skeleton.sum(), shape[axis], msg=axis)
#         nzc = thinning.kmath.list_of_tuples_of_val(cylinder_skeleton, 1)
#         prev_slope = np.subtract(nzc[0], nzc[1])
#         for p1, p2 in zip(nzc[:-1], nzc[1:]):
#             slope = np.subtract(p1, p2)
#             # all points lie on same straight line
#             np.testing.assert_array_equal(slope, prev_slope)
#             # all points are in the center of the cylinder
#             nose.tools.assert_in(15, p1)
#             nose.tools.assert_in(15, p2)
