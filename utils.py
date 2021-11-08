import imageio
import tifffile
import os


def read_img(filepath):
    extension = os.path.splitext(filepath)[1]
    if extension == '.tiff':
        img = tifffile.imread(filepath)
    else:
        img = imageio.imread(filepath)
    return img


def write_img(img, filepath):
    extension = os.path.splitext(filepath)[1]
    if extension == '.tiff':
        tifffile.imsave(filepath, img)
    else:
        imageio.imwrite(filepath, img)
