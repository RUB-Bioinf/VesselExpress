import time

import numpy as np
cimport cython

import skeleton.rotational_operators as rotational_operators
import skeleton.image_tools as image_tools
"""
cython convolve to speed up thinning
"""
with np.load(rotational_operators.LOOKUP_ARRAY_PATH)as lua:
    LOOKUP_ARRAY = lua["lua"]


SELEMENT = np.array([[[False, False, False], [False,  True, False], [False, False, False]],
                     [[False,  True, False], [True,  False,  True], [False,  True, False]],
                     [[False, False, False], [False,  True, False], [False, False, False]]], dtype=np.uint64)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cy_convolve(unsigned long long int[:, :, :] binary_arr, 
                unsigned long long int[:, :, :] kernel, 
                Py_ssize_t[:, ::1]  points, 
                str mode,
                int cval):
    """
    Returns convolved output only at points
    Parameters
    ----------
    binary_arr : Numpy array
        3D np.uint64 numpy array

    kernel : Numpy array
        3D flipped array of type uint64, shape (3, 3, 3) to convolve binary_arr with

    points : array
        array of 3D coordinates to find convolution at

    mode : string
        convolution mode, can be either 'constant' or 'reflect'
    cval : int
        value to pad with if mode is 'constant'
    Returns
    -------
    responses : Numpy array
        3D convolved numpy array only at "points"
    """
    cdef Py_ssize_t y, z, x, n, i, j, k
    cdef Py_ssize_t npoints = points.shape[0]
    cdef unsigned long long int[::1] responses = np.zeros(npoints, dtype='u8')
    z_extent, x_extent, y_extent = np.asarray(binary_arr).shape
    extent = [z_extent - 1, x_extent - 1, y_extent - 1]
    for n in range(npoints):
        z = points[n, 0]
        x = points[n, 1]
        y = points[n, 2]
        for k, i, j in rotational_operators.POSITION_VECTORS:
            if (z + k < 0 or z + k > extent[0] or 
                  x + i < 0 or x + i > extent[1] or 
                  y + j < 0 or y + j > extent[2]):
                if mode == 'reflect':
                    incremented = [z+k, x+i, y+j]
                    indexes = [0 if item < 0 else item for item in incremented]
                    indexes_new = [extent[nth] if value > extent[nth] else value for nth, value in enumerate(indexes)]
                    binary_arr_val = binary_arr[indexes_new[0], indexes_new[1], indexes_new[2]]
                elif mode == 'constant':
                    binary_arr_val = cval
            else:
                binary_arr_val = binary_arr[z + k, x + i, y + j]
            responses[n] += binary_arr_val * kernel[k + 1, i + 1, j + 1]
    return np.asarray(responses, order='C')


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_border_coords(unsigned long long int[:, :, :] arr, 
                      Py_ssize_t[:, ::1] points,
                      str mode,
                      int cval):
    """
    Returns convolved output only at points
    Parameters
    ----------
    arr : Numpy array
        3D binary numpy array of np.uint64

    points : array
        array of 3D coordinates to find borders at

    mode : string
        convolution mode, can be either 'constant' or 'reflect'

    cval : int
        value to pad with if mode is 'constant'
    Returns
    -------
    responses : Numpy array
        N * 3 array list of borders of objects in the arr
    """
    conv_arr = cy_convolve(arr, kernel=SELEMENT, points=points, mode=mode, cval=cval)
    border_point_arr_coordinates =  [index for value, index in zip(conv_arr, points) 
                                     if value != 6]
    return np.asarray(border_point_arr_coordinates, order='C')


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cy_get_thinned_3d(unsigned long long int[:, :, :] arr, str mode, int cval):
    """
    Return thinned output
    Parameters
    ----------
    binary_arr : Numpy array
        3D np.uint64 numpy array
    mode : string
        convolution mode, can be either 'constant' or 'reflect'
    cval : int
        value to pad with if mode is 'constant'
    Returns
    -------
    Numpy array
        3D np.bool thinned numpy array of the same shape
    Notes
    -----
    Nonzero point p is said to be a border point if the set N6(p)[1st orderd neighbors] contains at least one white point.
    In other words it is not a border point if every point is 1 i.e sum of the neighbors N6(p) = 6
    """
    cdef Py_ssize_t num_voxels_removed = 1
    cdef Py_ssize_t iter_count = 0
    cdef Py_ssize_t x, y, z
    # Loop until array doesn't change equivalent to you cant remove any pixels => num_voxels_removed = 0
    while num_voxels_removed > 0:
        # loop through all 12 subiterations
        iter_time = time.time()
        non_zero_coordinates = np.asarray(image_tools.list_of_tuples_of_val(np.asarray(arr), 1), order='C')
        num_voxels_removed = 0
        if non_zero_coordinates != []:
            border_point_arr_coordinates = get_border_coords(arr, non_zero_coordinates, mode, cval)
            if border_point_arr_coordinates != []:
                for i in range(12):
                    conf_volume = cy_convolve(arr, 
                                            kernel=rotational_operators.DIRECTIONS_LIST[i], 
                                            points=border_point_arr_coordinates,
                                            mode=mode,
                                            cval=cval)
                    for value, (x, y, z) in zip(conf_volume, border_point_arr_coordinates):
                        if LOOKUP_ARRAY[value]:
                            arr[x, y, z] = 0
                            num_voxels_removed += 1
        iter_count += 1
        print("Finished iteration %i, %0.2f s, removed %i pixels" % (iter_count, time.time() - iter_time, num_voxels_removed))
    return np.asarray(arr, dtype=np.bool)
