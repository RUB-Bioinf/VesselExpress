import time
import tifffile
from skimage.filters import threshold_li, frangi, threshold_local
from skimage.morphology import binary_closing, disk, remove_small_objects
from skimage.restoration import denoise_tv_chambolle
import os
import argparse
import matplotlib.pyplot as plt
from skimage import color

if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes Li Thresholding on image file of type .jpg')
    parser.add_argument('-i', type=str, help='input jpg image file to process')
    parser.add_argument('-ball_radius', type=int, default=3,
                        help='radius of ball structuring element for morphological closing')
    parser.add_argument('-artifact_size', type=int, default=5,
                        help='size of artifacts to be removed from the binary mask')
    args = parser.parse_args()

    img = plt.imread(args.i)
    input_file = os.path.abspath(args.i).replace('\\', '/')
    output_dir = os.path.dirname(input_file)

    img = color.rgb2gray(img)
    img = denoise_tv_chambolle(img, weight=0.9)
    filtered = frangi(image=img, black_ridges=False, sigmas=range(1, 20, 2), gamma=0.1, mode='reflect')

    #threshold = threshold_li(filtered)
    threshold = threshold_local(filtered, block_size=137)
    li = filtered > threshold

    segmented = binary_closing(li, disk(args.ball_radius))
    segmented = (remove_small_objects(segmented, args.artifact_size))

    tifffile.imsave(output_dir + '/Binary_' + os.path.basename(output_dir) + '.tif', (segmented * 255).astype('uint8'),
                    photometric='minisblack')

    print("Thresholding completed in %0.3f seconds" % (time.time() - programStart))
