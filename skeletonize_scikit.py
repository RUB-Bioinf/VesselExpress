import time
import os
import tifffile
import argparse
from skimage.morphology import skeletonize, binary_dilation

from create_stl import get_mesh

if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes scikit skeletonization on a binary image file of type .tif')
    parser.add_argument('-i', type=str, help='input tif image file to process')
    parser.add_argument('-pixel_dimensions', type=str, default="2.0,1.015625,1.015625",
                        help='Pixel dimensions in [z, y, x]')
    args = parser.parse_args()

    pixelDims = [float(item) for item in args.pixel_dimensions.split(',')]
    input_file = os.path.abspath(args.i).replace('\\', '/')
    output_dir = os.path.dirname(input_file)
    binArr = tifffile.imread(args.i)

    skel = skeletonize(binArr, method='lee')

    tifffile.imsave(output_dir + '/Skeleton_' + os.path.basename(output_dir) + '.tif', skel.astype('uint8'),
                    photometric='minisblack')

    if skel.ndim == 3:
        skel = binary_dilation(skel)
        skelMesh = get_mesh(skel, tuple(pixelDims))
        skelMesh.save(output_dir + '/Skeleton_' + os.path.basename(output_dir) + '.stl')

    print("Skeletonization completed in %0.3f seconds" % (time.time() - programStart))
