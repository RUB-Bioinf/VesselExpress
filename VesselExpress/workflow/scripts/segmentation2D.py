import time
from skimage.filters import frangi
from skimage.restoration import denoise_tv_chambolle
import os
import argparse
import numpy as np
from skimage import color
import sys
from skimage.filters import threshold_li, threshold_local
from skimage.morphology import binary_closing, ball, disk, remove_small_objects

# import modules
package = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'modules/'))
sys.path.append(package)

import utils

if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes Li Thresholding on image file of type .jpg')
    parser.add_argument('-i', type=str, help='input jpg image file to process')
    parser.add_argument('-sigma_min', type=float, default=1, help='Frangi sigma_min parameter')
    parser.add_argument('-sigma_max', type=float, default=10, help='Frangi sigma_max parameter')
    parser.add_argument('-sigma_steps', type=float, default=2, help='Frangi sigma_steps parameter')
    parser.add_argument('-alpha', type=float, default=0.5, help='Frangi alpha parameter')
    parser.add_argument('-beta', type=float, default=0.5, help='Frangi beta parameter')
    parser.add_argument('-gamma', type=float, default=15, help='Frangi gamma parameter')
    parser.add_argument('-denoise', type=int, default=1, help='set to 1 for prior denoising of image')
    parser.add_argument('-value', type=float, default=0,
                        help='set a value for thresholding, 0 for Li thresholding')
    parser.add_argument('-ball_radius', type=int, default=3,
                        help='radius of ball structuring element for morphological closing')
    parser.add_argument('-artifact_size', type=int, default=5,
                        help='size of artifacts to be removed from the binary mask')
    parser.add_argument('-block_size', type=int, default=137, help='block size for local thresholding, '
                                                                   '0 for global thresholding')
    parser.add_argument('-prints', type=bool, default=False, help='set to True to print runtime')
    args = parser.parse_args()

    img = utils.read_img(args.i)
    input_file = os.path.abspath(args.i).replace('\\', '/')
    output_dir = os.path.dirname(input_file)

    if os.path.splitext(args.i) != 'tiff':
        img = color.rgb2gray(img)

    if args.denoise == 1:
        img = denoise_tv_chambolle(img, weight=0.9)
    filtered = frangi(image=img, black_ridges=False, sigmas=np.arange(args.sigma_min, args.sigma_max, args.sigma_steps),
                      alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    filtered = np.round(filtered / (filtered.max() / 65535))

    # Thresholding with specific value
    if args.value != 0:
        threshold = args.value
    # Li thresholding
    else:
        if args.block_size == 0:
            threshold = threshold_li(img)  # global thresholding
        else:
            threshold = threshold_local(img, block_size=args.block_size)  # local thresholding
    thrImage = img > threshold

    # Binary closing
    binImage = binary_closing(thrImage, disk(args.ball_radius))

    # Artifact removal
    binImage = remove_small_objects(binImage, args.artifact_size)

    utils.write_img((binImage * 255).astype('uint8'), output_dir + '/Binary_' + os.path.basename(output_dir) + '.'
                    + input_file.split('.')[1])

    if args.prints:
        print("Segmentation completed in %0.3f seconds" % (time.time() - programStart))
