import time
from skimage.filters import frangi
from skimage.restoration import denoise_tv_chambolle
import os
import argparse
import numpy as np
from skimage import color
import sys

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
                      alpha=args.alpha, beta=args.beta, gamma=args.gamma, mode='reflect')
    filtered = np.round(filtered / (filtered.max() / 65535))

    utils.write_img(filtered.astype('uint16'), output_dir + '/Frangi_' + os.path.basename(output_dir) + '.'
                    + input_file.split('.')[1])

    if args.prints:
        print("Frangi completed in %0.3f seconds" % (time.time() - programStart))
