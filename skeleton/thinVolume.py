import time

import numpy as np
from skimage.morphology import skeletonize
# NOTE This does the pyx compilation of this extension
import pyximport; pyximport.install() # NOQA
import skeleton.thinning as thinning

"""
Thinning algorithm as described in
A Parallel 3D 12-Subiteration Thinning Algorithm Kálmán Palágyi,Graphical Models and Image Processing
Volume 61, Issue 4, July 1999, Pages 199-221 Attila Kuba, 1999
z is the nth image of the stack in 3D array and is the first dimension in this program
"""


def get_thinned(binaryArr, mode: str='reflect', cval=0):
    """
    Return thinned output
    Parameters
    ----------
    binaryArr : Numpy array
        2D or 3D binary numpy array

    Returns
    -------
    result : boolean Numpy array
        2D or 3D binary thinned numpy array of the same shape
    """
    assert np.max(binaryArr) in [0, 1], "input must always be a binary array"
    voxCount = np.sum(binaryArr)
    if voxCount == 0 or voxCount == binaryArr.size:
        return binaryArr
    elif len(binaryArr.shape) == 2:
        return skeletonize(binaryArr).astype(bool)
    else:
        start_time = time.time()
        # cast to uint64 to make configuration number calculation return the right range of values
        result = thinning.cy_get_thinned_3d(np.uint64(binaryArr), mode, cval)
        print(
            "thinned %i number of pixels in %0.2f seconds" % (voxCount, time.time() - start_time))
        return result


if __name__ == '__main__':
    sample = np.ones((5, 5, 5), dtype=np.uint8)
    resultSkel = get_thinned(sample)
