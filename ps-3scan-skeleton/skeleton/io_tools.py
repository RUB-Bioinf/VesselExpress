import glob
import inspect
import multiprocessing
import os
import warnings
from functools import partial


import numpy as np
from scipy.misc import imsave, imread
from scipy import ndimage
from PIL import Image

import skeleton.image_tools as image_tools


# Note: This disables the compression bomb warnings
# That PIL issues (over and over)
Image.MAX_IMAGE_PIXELS = None


def getFilePathList(path, targetExtension="png"):
    """
    Return a list of all files with target extension (default ".png") in the input path
    Excludes any hidden files (name starts with '.')
    """
    fileList = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    fileList = sorted(filter(lambda f: f.lower().endswith("." + targetExtension.lower()), fileList))
    return fileList


def saveStack(stack, path, prefix="", extension='png', displayProgress=True):
    """
    save stack as output images;
        if monochrome, expected dimensions are (x, y, z)
        if RGB, expected dimensions are (x, y, z, c)
    default extension is png
    Files are named by z index in stack, zero padded to 8 digits; eg "00000534.png"
    prefix is added to the start of each filename, with the default as the empty string ""
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    n = stack.shape[2]
    digits = 8
    isRGB = True if len(stack.shape) == 4 and stack.shape[3] == 3 else False

    for i in range(n):
        if displayProgress:
            print("saving image %i / %i \r" % (i + 1, n), end="", flush=True)
        fn = "{pre}{ix:0{w}d}.{e}".format(pre=prefix, ix=i, w=digits, e=extension)
        if type(stack[0, 0, 0]) is np.bool_:
            imsave(os.path.join(path, fn), stack[:, :, i] * 255)
        elif isRGB:
            imsave(os.path.join(path, fn), stack[:, :, i, :])
        else:  # monochrome
            imsave(os.path.join(path, fn), stack[:, :, i])
    if displayProgress:
        print("\n")


def loadStack(path, targetExtension="png", displayProgress=True):
    """
    given an input path, load all of the images sequentially from that directory
    and put them into a stack
    if monochrome, expected dimensions are (x, y, z)
    if RGB, expected dimensions are (x, y, z, c)
    """
    if not os.path.isdir(path):
        raise os.NotADirectoryError(path)

    fileList = getFilePathList(path, targetExtension)

    # load the first image to get image size
    tmpImg = imread(fileList[0])

    if image_tools.isRGB(tmpImg):
        stack = np.empty(tmpImg.shape[:2] + (len(fileList),) + (tmpImg.shape[2],), dtype=np.uint8)
    else:
        stack = np.empty(tmpImg.shape[:2] + (len(fileList),), dtype=np.uint8)

    for z, fn in enumerate(fileList):
        if image_tools.isRGB(tmpImg):
            stack[:, :, z, :] = imread(fn)
        else:
            stack[..., z] = imread(fn)
        if displayProgress:
            progress = int(100 * (z + 1) / len(fileList))
            print("loading volume from dir: {}% \r".format(progress), end="", flush=True)
    if displayProgress:
        print()
    return stack


def writeTransparentPngs(dirName, transparentValue=0):

    imNames = glob.glob(dirName + "*.png")
    for i in range(0, len(imNames)):
        im = imread(imNames[i])
        faceName = imNames[i].split("/")
        alphaDir = dirName + "alphaPngs/"
        if not os.path.exists(alphaDir):
            os.makedirs(alphaDir)

        writeTransparentPng(im, transparentValue=transparentValue, saveURI=alphaDir + faceName[-1])


def writeTransparentPng(im, transparentValue=0, saveURI=None):
    """
    Save image as a transparent png, turning background into aplha layers

    Iput parameters:
    im =               Input color image, uint8
    transparentValue = The intensity value that will become transparent alpha layer.
    saveURI =          The full path to where the image should be saved; e.g. "/home/image.png"

    Output parameters:
    im =               image with alpha channel added
    """

    # Make sure image is uint8
    im = im.astype(np.uint8)

    # Get the transparent mask, all pixels with transparentValue will be transparent
    pos = np.atleast_3d(255 * (im != transparentValue))
    # Gotta change this to 3D uint8 be able to run it through PIL
    pos = pos.astype(np.uint8)
    mask = Image.fromarray(pos)
    # Convert image too
    im = Image.fromarray(im)
    # Convert from RGB to grayscale
    mask = mask.convert("L")
    # Add the alpha channel to PIL image
    im.putalpha(mask)

    if saveURI is not None:
        # Make sure extension is png
        if (saveURI[-4:] != ".png"):
            saveURI = saveURI + ".png"
        # Write image
        im.save(saveURI)

    return im


def png2tif(source, destPath):
    """
    convert a single image to a tif at the desired dest
    """
    filepath, filename = os.path.split(source)
    fnbase, ext = os.path.splitext(filename)
    newFilename = os.path.join(destPath, fnbase + ".tif")
    # skip the file if it already exists
    # NOTE: this could be a problem if it is a half written file
    if not os.path.exists(newFilename):
        im = Image.open(source)
        im.save(newFilename)


def pngDir2tif(originPath, newPath, numProcesses=4):
    """
    convert a folder of png to a folder of tif, given target path
    parallel: use multiprocess to parallelize the save (default=True)
    showProgress: print out the progress of the filelist (default=False).
        Forced False if parallel=True
    """

    filelist = getFilePathList(originPath)
    if not os.path.exists(newPath):
        print("new directory not found, making directory:\n    {}".format(newPath))
        os.makedirs(newPath)
    nfile = len(filelist)
    func = partial(png2tif, destPath=newPath)
    with multiprocessing.get_context("spawn").Pool(processes=numProcesses) as pool:
        pool.map(func, filelist)

    print("Wrote {} TIF files to: {}".format(nfile, newPath))


def orthoSlice(stack, index, voxelSize=None, axis=2, order=0):
    """
    get the image slice from the stack at selected `index` and along desired `axis`
    If voxelSize is supplied, the image will be correctly interpolated
    axis=0 == YZ
    axis=1 == XZ
    axis=2 == XY
    NOTE: suggested to only be used on a Cubelet or smaller
    """
    stackSlice = [slice(None)] * len(stack.shape)
    stackSlice[axis] = index
    img = stack[stackSlice]
    if voxelSize:
        assert len(voxelSize) == 3 or len(voxelSize) == 4, "voxelSize should be a 3D or 4D tuple"
        zoomSize = [x for i, x in enumerate(voxelSize) if i != axis]
        img = ndimage.interpolation.zoom(img, zoomSize, order=order)
    if axis == 0:  # transpose if image is YZ
        img = img.T  # transpose so correct orientation
    return img


def saveOrthoStack(stack, path, voxelSize=None, axis=2, order=0):
    """
    given a 3d stack and a path, save images along 'axis' of stack
    """

    for ix in range(stack.shape[axis]):
        img = orthoSlice(stack, ix, voxelSize=voxelSize, axis=axis, order=order)
        fn = os.path.join(path, padInt(ix) + ".png")
        imsave(fn, img)
        print("saved image {}".format(fn))


def padInt(number, paddingDigits=8):
    """
    Return a string padded with a number of zeros specified
    - padInt(5, 2) -> "05"
    - padInt(5, 6) -< "000005"
    Raises assertion errors for non-integral types or negative values.
    """

    assert isinstance(number, int), "Number must be an integer, not {0}, of type: {1}".format(number, type(number))
    assert number >= 0, "Number must be semipositive, not {0}".format(number)
    if number >= 10 ** (paddingDigits + 1):
        warnings.warn("call to padInt() with insufficent digits to reproduce integer")

    return "{number:0{padding}d}".format(number=number, padding=paddingDigits)


def module_relative_path(path, *, depth=1):
    """
    Construct a path relative to the path of the calling module.
    'depth' - how far up the call stack to search for the module.
    """
    assert not path.startswith('/'), "Paths must be relative: {}".format(path)
    return os.path.abspath(os.path.join(module_dir(depth=depth + 1), path))


def module_dir(*, depth=1):
    """
    What is the source directory of the calling module?
    'depth' - how far up the call stack to search for the module.
    Ex: foo/kesm/bar/baz.py
    >>> import kesm.base
    >>> kesm.base.module_dir()
    'foo/kesm/bar'
    """
    filename = inspect.stack()[depth][1]
    return os.path.dirname(filename)

