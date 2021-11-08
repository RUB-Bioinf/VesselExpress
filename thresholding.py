import time
from skimage.filters import threshold_li, threshold_local
from skimage.morphology import binary_closing, ball, disk, remove_small_objects
import os
import argparse

import utils


if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes Li Thresholding on image file of type .tif')
    parser.add_argument('-i', type=str, help='input tif image file to process')
    parser.add_argument('-pixel_dimensions', type=str, default="2.0,1.015625,1.015625",
                        help='Pixel dimensions in [z, y, x] or [y, x]')
    parser.add_argument('-ball_radius', type=int, default=3,
                        help='radius of ball structuring element for morphological closing')
    parser.add_argument('-artifact_size', type=int, default=5,
                        help='size of artifacts to be removed from the binary mask')
    parser.add_argument('-block_size', type=int, default=137, help='block size for local thresholding')
    args = parser.parse_args()

    pixelDims = [float(item) for item in args.pixel_dimensions.split(',')]
    img = utils.read_img(args.i)
    input_file = os.path.abspath(args.i).replace('\\', '/')
    output_dir = os.path.dirname(input_file)

    start = time.time()

    if img.ndim == 3:
        dim3 = True
    else:
        dim3 = False

    # Li thresholding
    if dim3:
        threshold = threshold_li(img)                                   # global thresholding
    else:
        threshold = threshold_local(img, block_size=args.block_size)    # local thresholding
    thrImage = img > threshold

    # Binary closing
    if dim3:
        binImage = binary_closing(thrImage, ball(args.ball_radius))
    else:
        binImage = binary_closing(thrImage, disk(args.ball_radius))

    # Artifact removal
    binImage = remove_small_objects(binImage, args.artifact_size)

    utils.write_img((binImage * 255).astype('uint8'), output_dir + '/Binary_' + os.path.basename(output_dir) + '.'
                    + input_file.split('.')[1])

    print("Thresholding completed in %0.3f seconds" % (time.time() - programStart))
