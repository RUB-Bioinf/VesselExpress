import time
import tifffile
from skimage.filters import threshold_li
from skimage.morphology import binary_closing, ball, remove_small_objects
import os
import argparse
import numpy as np

if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes Li Thresholding on image file of type .tif')
    parser.add_argument('-i', type=str, help='input tif image file to process')
    parser.add_argument('-ball_radius', type=int, default=3,
                        help='radius of ball structuring element for morphological closing')
    parser.add_argument('-artifact_size', type=int, default=5,
                        help='size of artifacts to be removed from the binary mask')
    args = parser.parse_args()

    img = tifffile.imread(args.i)
    input_file = os.path.abspath(args.i).replace('\\', '/')
    output_dir = os.path.dirname(input_file)

    start = time.time()
    threshold = threshold_li(img)
    thrImage = img > threshold
    binImage = binary_closing(thrImage, ball(args.ball_radius))
    binImage = remove_small_objects(binImage, args.artifact_size)
    #print("binary foreground points=", np.count_nonzero(binImage), " for image ", input_file)
    print("elapsed time: %0.3f seconds" % (time.time() - start))

    tifffile.imsave(output_dir + '/Binary_' + os.path.basename(output_dir) + '.tif', (binImage * 255).astype('uint8'),
                    photometric='minisblack')

    print("Thresholding completed in %0.3f seconds" % (time.time() - programStart))
