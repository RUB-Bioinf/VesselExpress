import time
import os
import tifffile
import numpy as np
import argparse
from skimage.morphology import skeletonize

if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes scikit skeletonization on a binary image file of type .tif')
    parser.add_argument('-i', type=str, help='input tif image file to process')
    args = parser.parse_args()

    input_file = os.path.abspath(args.i).replace('\\', '/')
    output_dir = os.path.dirname(input_file)
    binArr = tifffile.imread(args.i)

    skel = skeletonize(binArr, method='lee')

    tifffile.imsave(output_dir + '/Skeleton_' + os.path.basename(output_dir) + '.tif', skel.astype('uint8'),
                    photometric='minisblack')

    print("Skeletonization completed in %0.3f seconds" % (time.time() - programStart))
