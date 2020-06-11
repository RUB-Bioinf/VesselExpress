import itertools
import os

import numpy as np

"""
This module has functions of operators to flip and rotate a cube
in 12 possible ways as in referenced in the paper

A Parallel 3D 12-_subiteration Thinning Algorithm Kálmán Palágyi,
Graphical Models and Image Processing Volume 61, Issue 4, July 1999,
Pages 199-221 Attila Kuba, 1999'

The cube must be of equal dimensions in x, y and z in this program
first dimension(0) is z, second(1) is y, third(2) is x

np.rot90 is not used to rotate an array by 90 degrees in the
counter-clockwise direction.

The first two dimensions are rotated; therefore, the array must be at least 2-D
"""

"""
TODO(crutcher, pranathi): The plane inversion for 'y' and 'z' is historical,
we probably want to flip it back when we can.
"""
ROTATIONAL_AXIS_MAP = {'x': [1, 2], 'y': [2, 0], 'z': [1, 0]}


def rot_3D_90(cube_array, rot_axis='z', k=0):
    """
    Returns a 3D array after rotating 90 degrees anticlockwise k times around rot_axis
    Parameters
    ----------
    cube_array : numpy array
        3D numpy array

    rot_axis : string
        can be z, y or x specifies which axis should the array be rotated around, by default 'z'

    k : integer
        indicates how many times to rotate, by default '0' (doesn't rotate)

    Returns
    -------
    rot_cube_array : array
        roated cube_array of the cube_array

    """
    assert 1 not in np.unique(cube_array.shape)
    assert cube_array.ndim == 3, "number of dimensions must be 3, it is %i " % cube_array.ndim
    return np.rot90(cube_array, k=k, axes=ROTATIONAL_AXIS_MAP[rot_axis])


def get_directions_list(cube_array):
    """
    Returns a list of rotated 3D arrays to change pixel to one of 12 directions
    where a border point should be removed according to Palagyi's thinning
    algorithm. the 12 directions are UN, UE, US, UW, NE, NW, ND, ES, ED, SW, SD, and WD
    Refer notes
    Arguments:
        cube_array : numpy array
            3D numpy array

    Returns:
        list

    Notes:
        UN - Up North, UE - Up East, US - Up South, UW - Up West, NE - North East
        NW - North West, ND - North Down, ES - East South, ED - East Down, SW - South West
        SD - South Down, WD - West Down
    """
    assert cube_array.ndim == 3, "number of dimensions must be 3, it is %i " % cube_array.ndim
    # mask outs border voxels in US
    first_subiteration = cube_array.copy(order='C')
    # mask outs border voxels in NE
    second_subiteration = rot_3D_90(rot_3D_90(cube_array, 'y', 2), 'x', 3).copy(order='C')
    # mask outs border voxels in WD
    third_subiteration = rot_3D_90(rot_3D_90(cube_array, 'x', 1), 'z', 1).copy(order='C')
    # mask outs border voxels in ES
    fourth_subiteration = rot_3D_90(cube_array, 'x', 3).copy(order='C')
    # mask outs border voxels in UW
    fifth_subiteration = rot_3D_90(cube_array, 'y', 3).copy(order='C')
    # mask outs border voxels in ND
    sixth_subiteration = rot_3D_90(rot_3D_90(rot_3D_90(cube_array, 'x', 3), 'z', 1), 'y', 1).copy(
        order='C')
    # mask outs border voxels in SW
    seventh_subiteration = rot_3D_90(cube_array, 'x', 1).copy(order='C')
    # mask outs border voxels in UN
    eighth_subiteration = rot_3D_90(cube_array, 'y', 2).copy(order='C')
    # mask outs border voxels in ED
    ninth_subiteration = rot_3D_90(rot_3D_90(cube_array, 'x', 3), 'z', 1).copy(order='C')
    # mask outs border voxels in NW
    tenth_subiteration = rot_3D_90(rot_3D_90(cube_array, 'y', 2), 'x', 1).copy(order='C')
    # mask outs border voxels in UE
    eleventh_subiteration = rot_3D_90(cube_array, 'y', 1).copy(order='C')
    # mask outs border voxels in SD
    twelveth_subiteration = rot_3D_90(cube_array, 'x', 2).copy(order='C')

    # List of 12 rotated configuration arrays
    DIRECTIONS_LIST = [
        first_subiteration, second_subiteration, third_subiteration, fourth_subiteration,
        fifth_subiteration, sixth_subiteration, seventh_subiteration, eighth_subiteration,
        ninth_subiteration, tenth_subiteration, eleventh_subiteration, twelveth_subiteration
    ]
    DIRECTIONS_LIST = [flip_convolve(direction) for direction in DIRECTIONS_LIST]
    return DIRECTIONS_LIST


def flip_convolve(kernel):
    return kernel[tuple([slice(None, None, -1)] * kernel.ndim)]


# Reference array is a flipped configuration number template we convolve thinning input with
# each of the 26 elements are represented in the 2nd order neighborhood are given a value of
# power(2, index)
# 26 configuration convolution kernel's elements list
REFERENCE_ARRAY = [2 ** i for i in range(0, 26)]
# at the center of 26 neighbors, insert 0
REFERENCE_ARRAY.insert(13, 0)
# Reshape it to a (3, 3, 3) matrix of uint64 to use it for convolution
REFERENCE_ARRAY = np.asarray(REFERENCE_ARRAY).reshape(3, 3, 3).astype(np.uint64)

# List of 12 functions corresponding to transformations in 12 directions
DIRECTIONS_LIST = get_directions_list(REFERENCE_ARRAY)

# Path of pre-generated lookuparray.npz
LOOKUP_ARRAY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lookuparray.npz')

POSITION_VECTORS = list(itertools.product((-1, 0, 1), repeat=3))
