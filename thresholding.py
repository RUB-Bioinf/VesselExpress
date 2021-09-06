import time
import tifffile
from skimage.filters import threshold_li
from skimage.morphology import binary_closing, ball, remove_small_objects
import os
import argparse
import numpy as np

from create_stl import get_mesh

if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes Li Thresholding on image file of type .tif')
    parser.add_argument('-i', type=str, help='input tif image file to process')
    parser.add_argument('-pixel_dimensions', type=str, default="2.0,1.015625,1.015625",
                        help='Pixel dimensions in [z, y, x]')
    parser.add_argument('-ball_radius', type=int, default=3,
                        help='radius of ball structuring element for morphological closing')
    parser.add_argument('-artifact_size', type=int, default=5,
                        help='size of artifacts to be removed from the binary mask')
    args = parser.parse_args()

    pixelDims = [float(item) for item in args.pixel_dimensions.split(',')]
    img = tifffile.imread(args.i)
    input_file = os.path.abspath(args.i).replace('\\', '/')
    output_dir = os.path.dirname(input_file)

    start = time.time()
    threshold = threshold_li(img)
    thrImage = img > threshold
    binImage = binary_closing(thrImage, ball(args.ball_radius))
    binImage = remove_small_objects(binImage, args.artifact_size)
    binMesh = get_mesh(binImage, tuple(pixelDims))
    #print("binary foreground points=", np.count_nonzero(binImage), " for image ", input_file)

    tifffile.imsave(output_dir + '/Binary_' + os.path.basename(output_dir) + '.tif', (binImage * 255).astype('uint8'),
                    photometric='minisblack')
    binMesh.save(output_dir + '/Binary_' + os.path.basename(output_dir) + '.stl')
    print("Thresholding completed in %0.3f seconds" % (time.time() - programStart))
