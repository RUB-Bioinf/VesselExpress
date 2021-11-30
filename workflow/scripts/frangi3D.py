import time
import argparse
import sys
import os

# import modules
package = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'modules/'))
sys.path.append(package)

from frangi import Frangi_filter

if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Computes Frangi filtering on image files of type .tif')
    parser.add_argument('-i', type=str, help='input tif image file to process')
    parser.add_argument('-sigma_min', type=int, default=2, help='Frangi sigma_min parameter')
    parser.add_argument('-sigma_max', type=int, default=5, help='Frangi sigma_max parameter')
    parser.add_argument('-sigma_steps', type=int, default=3, help='Frangi sigma_steps parameter')
    parser.add_argument('-alpha', type=float, default=0.5, help='Frangi alpha parameter')
    parser.add_argument('-beta', type=float, default=500.0, help='Frangi beta parameter')
    parser.add_argument('-gamma', type=float, default=200.0, help='Frangi gamma parameter')
    parser.add_argument('-prints', type=bool, default=False, help='set to True to print runtime')
    args = parser.parse_args()

    Frangi_filter(args.i, args.sigma_min, args.sigma_max, args.sigma_steps, args.alpha, args.beta, args.gamma)

    if args.prints:
        print("Frangi filtering completed in %0.3f seconds" % (time.time() - programStart))
