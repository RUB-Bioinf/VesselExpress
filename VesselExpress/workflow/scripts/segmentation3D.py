import numpy as np
from importlib import import_module
from skimage.morphology import remove_small_objects, binary_closing, cube
from aicssegmentation.core.utils import topology_preserving_thinning
from aicssegmentation.core.pre_processing_utils import edge_preserving_smoothing_3d
from tifffile import imread, imsave
import time
import argparse
from pathlib import Path
import itk
from typing import Union
from importlib import import_module
import os


def vesselness_filter(
    im: np.ndarray,
    dim: int,
    sigma: Union[int, float],
    cutoff_method: str
) -> np.ndarray:
    """
    function for running ITK 3D/2D vesselness filter

    Parameters:
    ------
    im: np.ndarray
        the 3D image to be applied on
    dim: int
        either apply 3D vesselness filter or apply 2D vesselness slice by slice
    sigma: Union[float, int]
        the kernal size of the filter
    cutoff_method: str
        which method to use for determining the cutoff value, options include any
        threshold method in skimage, such as "threshold_li", "threshold_otsu",
        "threshold_triangle", etc.. See https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html

    Returns:
    ---------
    vess: np.ndarray
        filter output
    """
    if dim == 3:
        im_itk = itk.image_view_from_array(im)
        hessian_itk = itk.hessian_recursive_gaussian_image_filter(im_itk, sigma=sigma, normalize_across_scale=True)
        vess_tubulness = itk.hessian_to_objectness_measure_image_filter(hessian_itk, object_dimension=1)
        vess = np.asarray(vess_tubulness)
    elif dim ==2:
        vess = np.zeros_like(im)
        for z in range(im.shape[0]):
            im_itk = itk.image_view_from_array(im[z,:,:])
            hessian_itk = itk.hessian_recursive_gaussian_image_filter(im_itk, sigma=sigma, normalize_across_scale=True)
            vess_tubulness = itk.hessian_to_objectness_measure_image_filter(hessian_itk, object_dimension=1)
            vess_2d = np.asarray(vess_tubulness)
            vess[z, :, :] = vess_2d[:, :]

    module_name = import_module("skimage.filters")
    threshold_function = getattr(module_name, cutoff_method)

    return vess > threshold_function(vess)


def threshold_by_variation(im: np.ndarray, scale: Union[int, float]) -> np.ndarray:
    """
    thresholding method based on basic statistics. The threshold is calcualted
    as "mean intensity of the image intensity" + `scale` * "the standard deviation of
    the image intensity".

    Parameters:
    ----------
    ------
    im: np.ndarray
        the 3D image to be applied on
    scale: Union[float, int]
        how many fold of the standard deviation of the image intensity will be used
        to calculat the threshold

    Returns:
    ---------
    thresh: float
        the threshold value
    """

    thresh = im.mean() + scale * im.std()
    return im > thresh


def segmentation(input, output, cfg):
    im = imread(input)

    #######################
    # pre-processing
    #######################
    if cfg["smoothing"] == 1:
        im = edge_preserving_smoothing_3d(im)

    #######################
    # core steps
    #######################
    if cfg["core_threshold"] is not None:
        seg = threshold_by_variation(im, cfg["core_threshold"])
    else:
        seg = np.zeros_like(im) > 0

    if cfg["core_vessel_1"] == 1:
        out = vesselness_filter(im, cfg["dim_1"], cfg["sigma_1"], cfg["cutoff_method_1"])
        seg = np.logical_or(seg, out)

    if cfg["core_vessel_2"] == 1:
        out = vesselness_filter(im, cfg["dim_2"], cfg["sigma_2"], cfg["cutoff_method_2"])
        seg = np.logical_or(seg, out)

    #######################
    # post-processing
    #######################
    if cfg["post_closing"] is not None:
        seg = binary_closing(seg, cube(cfg["post_closing"]))

    if cfg["post_thinning"] == 1:
        seg = topology_preserving_thinning(seg, cfg["post_thinning"])

    if cfg["post_cleaning"] is not None:
        seg = remove_small_objects(seg, min_size=cfg["post_cleaning"])

    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255

    imsave(output, seg)


if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='segmentation of 3D images of type .tif')
    parser.add_argument('-input', type=str, help='input of raw tif image')
    parser.add_argument('-prints', type=bool, default=False, help='set to True to print runtime')
    parser.add_argument('-smoothing', type=int, default=1, help='set to 1 for smoothing, 0 for no smoothing')
    parser.add_argument('-core_threshold', type=float, default=3.0)
    parser.add_argument('-core_vessel_1', type=int, default=1, help='set to 1 for first vesselness filter, '
                                                                    '0 for no vesselness filter')
    parser.add_argument('-dim_1', type=int, default=3)
    parser.add_argument('-sigma_1', type=float, default=1.0)
    parser.add_argument('-cutoff_method_1', type=str, default='threshold_li')
    parser.add_argument('-core_vessel_2', type=int, default=1, help='set to 1 for second vesselness filter, '
                                                                    '0 for no vesselness filter')
    parser.add_argument('-dim_2', type=int, default=3)
    parser.add_argument('-sigma_2', type=float, default=2.0)
    parser.add_argument('-cutoff_method_2', type=str, default='threshold_li')
    parser.add_argument('-post_closing', type=int, default=5)
    parser.add_argument('-post_thinning', type=int, default=0, help='set to 1 for post-processing, '
                                                                    '0 for no post-processing')
    parser.add_argument('-min_thickness', type=int, default=None)
    parser.add_argument('-thin', type=int, default=None)
    parser.add_argument('-post_cleaning', type=int, default=100)
    args = parser.parse_args()

    config = {
        "smoothing": args.smoothing,
        "core_threshold": args.core_threshold,
        "core_vessel_1": args.core_vessel_1,
        "dim_1": args.dim_1,
        "sigma_1": args.sigma_1,
        "cutoff_method_1": args.cutoff_method_1,
        "core_vessel_2": args.core_vessel_2,
        "dim_2": args.dim_2,
        "sigma_2": args.sigma_2,
        "cutoff_method_2": args.cutoff_method_2,
        "post_closing": args.post_closing,
        "post_thinning": args.post_thinning,
        "min_thickness": args.min_thickness,
        "thin": args.thin,
        "post_cleaning": args.post_cleaning
    }

    input_file = os.path.abspath(args.input).replace('\\', '/')
    output_dir = os.path.dirname(input_file)

    segmentation(args.input, output_dir + '/Binary_' + os.path.basename(output_dir) + '.' + input_file.split('.')[1],
                 config)

    if args.prints:
        print("Segmentation completed in %0.3f seconds" % (time.time() - programStart))
