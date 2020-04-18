"""
This file should be a library of basic image processing tools.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.transform import downscale_local_mean
from sklearn.decomposition import PCA, FastICA


def isRGB(img):
    """
    return True if image is >=3 dimensions and the 3rd dimension is 3
    """
    return len(img.shape) >= 3 and img.shape[2] == 3


def isRGBStack(im):
    """
    return True if image is >=3 dimensions and the 3rd dimension is 3
    """
    return im.ndim == 4 and im.shape[2] == 3


def dtypeMinMax(img):
    """
    returns the max and min for a datatype in an image
    if float, sets to (0, 1), else sets to range for datatype
    """
    if img.dtype in [float, np.float16, np.float32, np.float64]:
        return (0., 1.)
    else:
        ii = np.iinfo(img.dtype)
        return (ii.min, ii.max)


def int2Float(img, dtype=np.float32):
    """
    take in an N-D image of any datatype, and rescale it to [0, 1] inclusive
    NOTE: This does NOT scale the max and min values to match 0 and 1, it only fits a 0-255 image into 0-1
    """
    assert type(img) is np.ndarray
    ii = np.iinfo(img.dtype)
    return (1 / (ii.max - ii.min)) * (img - ii.min).astype(dtype)


def imgNorm(img, dtype=np.float32):
    """
    converts any image dtype to a float between 0 and 1 inclusive
    sets image min() to 0, image max() to 1
    default output dtype is float32
    """
    return (1 / (img.max() - img.min())) * (img - img.min()).astype(dtype)


def img255(img):
    """
    return an np.uint8 image from a float [0,1]
    """
    out = img.copy()
    np.multiply(out, 255, out=out)
    np.around(out, out=out)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def img2Bool(img):
    """
    Given a np.uint8 with 0 and 255 as the boolean values, reduce back to a bool
    This will take any nonzero value and turn it to a True
    """
    return img.astype(np.bool)


def scaleBool(bimg):
    """
    given a binary image containing only 1s and zeros or true and false, scale to min and max of uint8
    """
    return bimg.astype(np.uint8) * 255


def gray2RGB(img):
    """
    take in a MxN 2D grayscale image and return a MxNx3 grayscale image
    """
    return img[..., None] * [1, 1, 1]


def RGB2Gray(im):
    """
    Convert a single or stack of RGB images to greyscale image
    Input
    im        =  color iamge or stack of color images of shape M x N x 3, M x N x 3 x Z
    Output
    image of shape M x N or M x N x Z image with values between [0, 255]
    """
    if isRGB(im):
        return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    elif isRGBStack(im):
        greyScalestack = np.zeros((im.shape[0], im.shape[1], im.shape[3]), dtype=np.uint8)
        for i in range(im.shape[3]):
            greyScalestack[:, :, i] = RGB2Gray(im[:, :, :, i])
        return greyScalestack
    else:
        assert False, "given input is not a stack of RGB images or RGB image"


def RGB2Binary(im, threshold=127):
    """
    Convert a single or stack of RGB images to Binary image
    Input
    im        =  color iamge or stack of color images of shape M x N x 3, M x N x 3 x Z
    threshold = uint8 value threshold; by default it applies a binary threshold of 127 on grey scale image
    Output
    image of shape M x N or M x N x Z image with values between [0, 1] or boolean
    """
    maxVal = 1
    if isRGB(im):
        retThresh, binImg = cv2.threshold(RGB2Gray(im), threshold, maxVal, cv2.THRESH_BINARY)
        return binImg
    elif isRGBStack(im):
        binaryStack = np.zeros((im.shape[0], im.shape[1], im.shape[3]))
        for i in range(im.shape[3]):
            binaryStack[:, :, i] = RGB2Binary(im[:, :, i])
        return binaryStack
    else:
        assert False, "given input is not a stack of RGB images or RGB image"


def smoothIm(im, kernelSize=5):
    return cv2.GaussianBlur(im, (kernelSize, kernelSize), 0)


def backgroundSubtract(img, bg_value=None):

    """
    subtract the background value = (weighted mean intensity) from the image
    weighted mean = xp(x) where p(x) = p(occurence of x) / sum(possible values of various x occured)
    background subtraction can be done to decrease the influence of background when obtaining
    the foreground
    thresholding on the values of histogram other than zeros will give more accurate result
    img[img < int(bg_value)] = 0
    remArray = [y for y in img.ravel() if y != 0]
    t = threshold_otsu(np.array(remArray))
    """
    if bg_value is None:
        histData, bins = np.histogram(img, bins=range(257), density=True)
        bg_value = np.average(bins[:-1], weights=histData)
    return (img - bg_value).round().clip(0, 255).astype(np.uint8)


def colorRangeMask(img, color1, color2=[255, 255, 255]):
    """
    return binary mask of img.shape of all pixels with values between color1 and color2

    Input
    img    = color image M X N X 3
    color1 = [R, G, B] indicative of the lower limit of color desired
    color2 = [R, G, B] indicative of the upper limit of color desired
    Output
    Boolean array of shape img.shape with True where the color is between color1 and color2

    Example:
    mask = colorRangeMask(img, [135, 150, 0], [165, 165, 80])
    cleanImg = img * np.invert(mask[:,:, None])
    """
    assert isRGB(img), "input must be color image"
    maskR, maskG, maskB = [np.logical_and(img[:, :, i] <= color2[i], img[:, :, i] >= color1[i]) for i in range(3)]
    return np.logical_and(np.logical_and(maskR, maskG), maskB)


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


def maximumIntensityProjection(stack, axis=2, voxelSize=None):
    """
    return maximum intensity projection in 2D from image stack
    Stretches out Z according to the voxel aspect ratio
    Assumes that objects are in white, background is black

    axis does the projection along the desired axis [default:2 <z>]
    voxelSize is a tuple of the voxel aspect ratio (in UM is fine)

    NOTE: not written for color yet!
    NOTE: suggested to only be used on a Cubelet or smaller
    """
    assert len(stack.shape) == 3, "Only works for grayscale images so far"
    maxProj = np.max(stack, axis=axis)
    if axis == 0:  # transpose the YZ projection
        maxProj = maxProj.T
    if voxelSize:
        assert len(voxelSize) == 3, "voxelSize should be a 3D tuple"
        zoomSize = [x for i, x in enumerate(voxelSize) if i != axis]
        maxProj = ndimage.interpolation.zoom(maxProj, zoomSize, order=0)
    return maxProj


def alphaProjection(stack, n=10, alpha=0.95, bgSubtract=False):
    """
    do an alpha projection for n faces within a stack, decreasing the luminance by alpha
    NOTE: Doesn't seem to work so well with binary images.
    NOTE: suggested to only be used on a Cubelet or smaller
    """
    if stack.dtype == bool:
        stack = stack.astype(np.uint8) * 255
    alphaProj = stack[:, :, 0, ...].astype(np.uint32)
    d = 1 - alpha
    for i in range(n - 1, -1, -1):
        img = stack[:, :, i, ...]
        if bgSubtract:
            bg = np.median(img)
            img = backgroundSubtract(img, bg)
        alphaProj += (img * (1 - d * i)).astype(np.uint32)
    alphaProj = img255(alphaProj / np.max(alphaProj))
    return alphaProj


def sk_downscale(imageIn, factors):
    """
    use the scikit-image method for downsampling
    needs a tuple with same number of factors as dim of imgIn
    """
    return downscale_local_mean(imageIn, factors)


def projectImagePCA(im, normalize=True, whiten=False):
    """
    The function takes a color image and projects it onto its principal components after removing background pixels.

    Input parameters:
    im =           color image with background pixels set to zero
    normalize =    If true, normalize image to [0, 255] and convert to uint8 after PCA
    whiten =       If true, whitens the data before PCA

    Output parameters:
    imPCA =        3 channel image, each channel contains principal component projections
    eigenValues =  PCA eigenvalues
    eigenVectors = PCA eigenvectors

    Created by: JMF
    """
    # Assert color image
    assert isRGB(im), "Image must be 3 channel color image"

    featVect = im.reshape((-1, 3))
    featVect = featVect.astype(np.float32)
    labelIm = np.zeros_like(featVect)

    imGray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # Only perform principal component analysis on non-zero pixels.
    imGray = 1 * (imGray > 0)
    featBG = imGray.reshape((-1, 1))
    # Perform PCA on foreground pixels only
    pos = np.where(featBG == 0)[0]
    featVect = np.delete(featVect, pos, axis=0)
    # PCA and projection on axes
    pca = PCA(whiten=whiten)
    Spca = pca.fit(featVect).transform(featVect)
    # Reconstruct image
    pos = np.where(featBG > 0)[0]
    labelIm[pos, :] = Spca
    imPCA = labelIm.reshape(im.shape)
    # Normalize images
    if normalize:
        for i in range(3):
            PCAprojection = imPCA[:, :, i]
            imPCA[:, :, i] = (PCAprojection - PCAprojection.min()) / (PCAprojection.max() - PCAprojection.min()) * 255
        # Convert back to uint8
        imPCA = imPCA.astype(np.uint8)

    return imPCA, pca.explained_variance_ratio_, pca.components_


def projectImageICA(im, normalize=True):
    """
    The function takes a color image and projects it onto its independent components after removing background pixels.

    Input parameters:
    im =        color image with background pixels set to zero
    normalize = If true, normalize image to [0, 255] and convert to uint8

    Output parameters:
    imPCA =     3 channel image, each channel contains independent component projections
    unmixMat =  ICA unmixing matrix

    Created by: JMF
    """
    # Assert color image
    assert isRGB(im), "Image must be 3 channel color image"

    featVect = im.reshape((-1, 3))
    featVect = featVect.astype(np.float32)
    labelIm = np.zeros_like(featVect)

    imGray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    # Only perform principal component analysis on non-zero pixels.
    imGray = 1 * (imGray > 0)
    featBG = imGray.reshape((-1, 1))
    # Perform ICA on foreground pixels only
    pos = np.where(featBG == 0)[0]
    featVect = np.delete(featVect, pos, axis=0)
    # ICA and projection on axes
    ica = FastICA()
    Sica = ica.fit(featVect).transform(featVect)
    # Reconstruct image
    pos = np.where(featBG > 0)[0]
    labelIm[pos, :] = Sica
    imICA = labelIm.reshape(im.shape)
    # Normalize images
    if normalize:
        for i in range(3):
            ICAprojection = imICA[:, :, i]
            imICA[:, :, i] = (ICAprojection - ICAprojection.min()) / (ICAprojection.max() - ICAprojection.min()) * 255
        # Convert back to uint8
        imICA = imICA.astype(np.uint8)

    return imICA, ica.components_


def list_of_tuples_of_val(arr, value=0):
    """
    Returns list of tupled indices at which the arr is equal to value.
    Input:
        array: ndarray
    Returns:
        list of tuples at which arr is equal to value
    """
    return list(map(tuple, np.transpose(np.where(arr == value))))
