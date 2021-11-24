import os
import time
import argparse

from FrangiNet.module import main_FN_test

if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='Use FrangiNet to filter on image files of type .tif')
    parser.add_argument('-i', type=str, help='input tif image file to process')
    parser.add_argument('-o', type=str, help='output tif image file to process')
    parser.add_argument('-model', type=str, help='trained model to use for filtering')
    parser.add_argument('-mode', type=str, help='mode of the trained model')
    parser.add_argument('-normalization', type=str, help='Normalize image to range 0-1')
    parser.add_argument('-average', type=str, help='if true the average of 3D is taken, else the maximum')
    parser.add_argument('-mode_img', type=str, help='mode how the image is being processed')
    parser.add_argument('-gpus', type=str, help='number of the gpus that shall be used')
    parser.add_argument('-batch_size', type=str, help='only mode OneCubeBatch: size of the batch to be processed')
    args = parser.parse_args()

    main_FN_test(args.i, args.o, args.model, args.mode, args.normalization, args.average, args.mode_img, args.gpus,
                 args.batch_size)

    print("Frangi filtering completed in %0.3f seconds" % (time.time() - programStart))

    f = open(os.path.dirname(os.path.abspath(args.i).replace('\\', '/')) + '/measurement.csv', 'a')
    f.write("Frangi-Filter;" + "{:.3f}".format(time.time() - programStart) + ";s\n")
    f.close()
