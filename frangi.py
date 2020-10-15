import time
import os
import glob
import shutil
import multiprocessing as mp
from itertools import repeat
import argparse

#from frangi import Frangi_filter
from Segmentation.frangi import Frangi_filter

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
    args = parser.parse_args()

    Frangi_filter(args.i, args.sigma_min, args.sigma_max, args.sigma_steps, args.alpha, args.beta, args.gamma)

    print("Frangi filtering completed in %0.3f seconds" % (time.time() - programStart))
