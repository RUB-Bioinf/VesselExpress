import functools

import numpy as np

from skeleton.rotational_operators import get_directions_list

"""
Following is an application of memoization - pre-generating a look up array.
In this algorithm the look up array is of length (2 ** 26)
that has all possible configurations of binary strings of length 26 representing
26 voxels around a voxel at origin in a cube as indices and values as
a boolean that says the voxel can be removed or not (as it belongs to the boundary
not the skeleton) as in reference paper
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
"""


class Templates:
    def __init__(self, *args):
        (self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h, self.i, self.j, self.k, self.l, self.m, self.n,
         self.o, self.p, self.q, self.r, self.s, self.t, self.u, self.v, self.w, self.x, self.y, self.z) = [arg for arg
                                                                                                            in args]

    def first_template(self):
        result = ((not self.a) & (not self.b) & (not self.c) & (not self.j) & (not self.k) & (not self.l) &
                  (not self.r) & (not self.s) & (not self.t) & self.p &
                  (self.d | self.e | self.f | self.m | self.n | self.u | self.v | self.w | self.g | self.h | self.i |
                   self.o | self.q | self.x | self.y | self.z))
        return result

    def second_template(self):
        result = ((not self.a) & (not self.b) & (not self.c) & (not self.d) & (not self.e) & (not self.f) &
                  (not self.g) & (not self.h) & (not self.i) & self.v &
                  (self.r | self.s | self.t | self.j | self.k | self.l | self.m | self.n | self.u | self.w |
                   self.o | self.p | self.q | self.x | self.y | self.z))
        return result

    def third_template(self):
        result = ((not self.a) & (not self.b) & (not self.c) & (not self.j) & (not self.k) & (not self.l) &
                  (not self.r) & (not self.s) & (not self.t) & (not self.d) &
                  (not self.e) & (not self.f) & (not self.g) & (not self.h) & (not self.i) & self.y & (self.m | self.n |
                  self.u | self.w | self.o | self.q | self.x | self.z))
        return result

    def fourth_template(self):
        result = ((not self.a) & (not self.b) & (not self.c) & (not self.k) & (not self.e) & (not (self.d & self.j)) &
                  (not (self.l & self.f)) & self.p & self.v)
        return result

    def fifth_template(self):
        result = ((not self.a) & (not self.b) & (not self.k) & (not self.e) & self.c & self.v &
                  self.p & (not (self.j & self.d)) & (self.l ^ self.f))
        return result

    def sixth_template(self):
        result = (self.a & self.v & self.p & (not self.b) & (not self.c) & (not self.k) & (not self.e) &
                  (not (self.l & self.f)) & (self.j ^ self.d))
        return result

    def seventh_template(self):
        result = ((not self.a) & (not self.b) & (not self.k) & (not self.e) & self.n & self.v & self.p &
                  (not (self.j & self.d)))
        return result

    def eighth_template(self):
        result = ((not self.b) & (not self.c) & (not self.k) & (not self.e) & self.m & self.v & self.p &
                  (not (self.l & self.f)))
        return result

    def ninth_template(self):
        result = ((not self.b) & (not self.k) & (not self.e) & self.a & self.n & self.v & self.p & (self.j ^ self.d))
        return result

    def tenth_template(self):
        result = ((not self.b) & (not self.k) & (not self.e) & self.c & self.m & self.v & self.p & (self.l ^ self.f))
        return result

    def eleventh_template(self):
        result = ((not self.a) & (not self.b) & (not self.c) & (not self.j) & (not self.k) & (not self.l) &
                  (not self.r) & (not self.s) & (not self.t) & (not self.d) &
                  (not self.e) & (not self.g) & (not self.h) & self.q & self.y)
        return result

    def twelveth_template(self):
        result = ((not self.a) & (not self.b) & (not self.c) & (not self.j) & (not self.k) & (not self.l) &
                  (not self.r) & (not self.s) & (not self.t) & (not self.e) &
                  (not self.f) & (not self.h) & (not self.i) & self.o & self.y)
        return result

    def thirteenth_template(self):
        result = ((not self.a) & (not self.b) & (not self.c) & (not self.j) & (not self.k) & (not self.r) &
                  (not self.s) & (not self.d) & (not self.e) & (not self.f) &
                  (not self.g) & (not self.h) & (not self.i) & self.w & self.y)
        return result

    def fourteenth_template(self):
        result = ((not self.a) & (not self.b) & (not self.c) & (not self.d) & (not self.e) & (not self.f) &
                  (not self.g) & (not self.h) & (not self.i) &
                  (not self.k) & (not self.l) & (not self.s) & (not self.t) & self.u & self.y)
        return result


def _get_voxel_deletion_flag(neighbor_values, direction):
    """
    Returns self.a flag saying self.voxel should be deleted or not

    Parameters
    ----------
    neighbor_values : list
        list of first order neighborhood (26 voxels) of a nonzero value at origin

    direction : array
       transformation array describing rotation of cube to remove boundary voxels in a different direction

    Returns
    -------
    should_voxel_be_deleted : boolean
        0 => should not be deleted
        1 => should be deleted

    """
    assert len(neighbor_values) == 27
    # reshape neighbor_values to a 3 x 3 x 3 cube
    neighbor_matrix = np.reshape(neighbor_values, (3, 3, 3))
    # transform neighbor_values to direction
    neighbor_values = get_directions_list(neighbor_matrix)[direction]
    neighbor_values = list(np.reshape(neighbor_values, 27))
    del(neighbor_values[13])
    # assign 26 voxels in a 2nd ordered neighborhood of a 3D voxels as 26 alphabet variables
    neighbor_values = tuple(neighbor_values)
    # insert aplhabetical variables into equations of templates for deleting the boundary voxel
    template = Templates(*neighbor_values)
    should_voxel_be_deleted = functools.reduce(lambda x, y: x | y, [template.first_template(),
                                                                    template.second_template(),
                                                                    template.third_template(),
                                                                    template.fourth_template(),
                                                                    template.fifth_template(),
                                                                    template.sixth_template(),
                                                                    template.seventh_template(),
                                                                    template.eighth_template(),
                                                                    template.ninth_template(),
                                                                    template.tenth_template(),
                                                                    template.eleventh_template(),
                                                                    template.twelveth_template(),
                                                                    template.thirteenth_template(),
                                                                    template.fourteenth_template()])
    return should_voxel_be_deleted


def generate_lookup_array(stop=2**26, direction=0):
    """
    Returns lookuparray

    Parameters
    ----------
    stop : int
    integer describing the length of array

    direction : int
       describing nth rotation of cube to remove boundary voxels in a different direction

    Returns
    -------
    lookup_array : array
        value at an index of the array = 0 => should not be deleted
        value at an index of the array = 1 => should be deleted

    Notes
    ------
    This program is run once, and the array is saved as lookuparray.npy in
    the same folder in main function. It doesn't have to be run again unless if templates are changed

    """
    lookup_array = np.zeros(stop, dtype=bool)
    for item in range(0, stop):
        # convert the decimal number to a binary string
        neighbor_values = [(item >> digit) & 0x01 for digit in range(26)]
        # if it's a single non zero voxel in the 26 neighbors
        if np.sum(neighbor_values) == 1:
            lookup_array[item] = 0
        else:
            # voxel at origin/center of the cube should be nonzero, so insert
            neighbor_values.insert(13, 1)
            lookup_array[item] = _get_voxel_deletion_flag(neighbor_values, direction)
    return lookup_array


if __name__ == '__main__':
    # generating and saving all the 12 lookuparrays
    for index in range(12):
        lookup_array = generate_lookup_array(2 ** 26, index)
        np.save("lookuparray%i.npy" % (index + 1), lookup_array)
