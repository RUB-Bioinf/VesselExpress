from tifffile import imsave, imread
import numpy as np
from skimage.morphology import ball, remove_small_objects, binary_closing, cube
from skimage.filters import threshold_otsu, threshold_li, threshold_triangle
import time
import argparse
import os
import sys

# import modules
package = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'modules/'))
sys.path.append(package)

import utils


def segmentation(raw, small_frangi, large_frangi, output):
    im = utils.read_img(raw)
    chunk_th = (threshold_li(im) + threshold_otsu(im)) / 2
    chunk = im > chunk_th

    im2 = utils.read_img(small_frangi)
    seg2 = im2 > (threshold_li(im2) + threshold_triangle(im2)) / 2

    im3 = utils.read_img(large_frangi)
    seg3 = im3 > (threshold_li(im3) + threshold_otsu(im3)) / 2

    part2 = np.logical_or(seg2, seg3)

    merged_seg = np.logical_or(chunk, part2)

    final_seg = remove_small_objects(
        binary_closing(merged_seg, cube(3)), min_size=50
    )

    final_seg = final_seg.astype("uint8")
    final_seg[final_seg > 0] = 255

    utils.write_img(final_seg, output)


if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Combines 2 frangi filtered images to one segmentation mask')
    parser.add_argument('-raw', type=str, help='raw tif image')
    parser.add_argument('-frangi_small', type=str, help='frangi filtered tif image with small sigma')
    parser.add_argument('-frangi_large', type=str, help='frangi filtered tif image with large sigma')
    parser.add_argument('-prints', type=bool, default=False, help='set to True to print runtime')
    args = parser.parse_args()

    input_file = os.path.abspath(args.raw).replace('\\', '/')
    output_dir = os.path.dirname(input_file)
    output_file = output_dir + '/Binary_' + os.path.basename(output_dir) + '.' + input_file.split('.')[1]

    segmentation(args.raw, args.frangi_small, args.frangi_large, output_file)

    if args.prints:
        print("Thresholding completed in %0.3f seconds" % (time.time() - programStart))