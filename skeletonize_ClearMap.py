import argparse
import numpy as np
import os

import time
import ClearMap.IO.IO as io
import ClearMap.ImageProcessing.Skeletonization.Skeletonization as skl


if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes ClearMap skeletonization on a binary image file of type .tif')
    parser.add_argument('-i', type=str, help='input tif image file to process')
    args = parser.parse_args()

    input_file = os.path.abspath(args.i).replace('\\', '/')
    output_dir = os.path.dirname(input_file)
    file_name = os.path.basename(output_dir)

    seg = io.read(args.i)
    seg = seg / seg.max()
    np.save(output_dir + '/Binary_' + file_name + '.npy', seg)

    print("Skeletonization")
    start = time.time()
    sink_skel = output_dir + "/Skeleton_" + file_name
    im_skel = skl.skeletonize(output_dir + "/Binary_" + file_name + '.npy', sink=sink_skel + '.npy', delete_border=True,
                              verbose=False)
    print("elapsed time: %0.3f seconds" % (time.time() - start))

    im_skel.array = im_skel.array.astype(np.uint8) * 255
    io.write(sink_skel + ".tif", im_skel)

    print("Skeletonization completed in %0.3f seconds" % (time.time() - programStart))
