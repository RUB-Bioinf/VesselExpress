import numpy as np
from numpy import array
from importlib import import_module
from skimage.morphology import remove_small_objects, binary_closing, cube
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
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
import uuid
import dask.array as da
from dask.array import logical_or
from dask_image import ndmeasure, ndmorph
from functools import partial
from shutil import rmtree


def vesselness_filter(
        im: np.ndarray,
        sigma: Union[int, float] = 1,
        gamma: Union[int, float] = 5,
        cutoff_method: str = None,
        dim: int = 3,
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
    gamma: Union[float, int]
        the sensitivity of the filter to tubular shapes
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
        vess_tubulness = itk.hessian_to_objectness_measure_image_filter(hessian_itk, object_dimension=1, gamma=gamma)
        vess = np.asarray(vess_tubulness)
    elif dim == 2:
        vess = np.zeros_like(im)
        for z in range(im.shape[0]):
            im_itk = itk.image_view_from_array(im[z, :, :])
            hessian_itk = itk.hessian_recursive_gaussian_image_filter(im_itk, sigma=sigma, normalize_across_scale=True)
            vess_tubulness = itk.hessian_to_objectness_measure_image_filter(hessian_itk, object_dimension=1,
                                                                            gamma=gamma)
            vess_2d = np.asarray(vess_tubulness)
            vess[z, :, :] = vess_2d[:, :]

    if cutoff_method is None:
        return vess
    else:
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


class vesselness_filter_dask:
    def __init__(self, sigma, gamma):
        self.sigma = sigma
        self.gamma = gamma

    def compute_vesselness(self, im_smooth):
        return vesselness_filter(im_smooth.astype(float), sigma=self.sigma, gamma=self.gamma)


class dask_threshold_calculator:
    def __init__(self, cutoff_method):
        module_name = import_module("skimage.filters")
        self.threshold_function = getattr(module_name, cutoff_method)

    def calculate_by_chunks(self, im):
        th = self.threshold_function(im)
        return array(th)[None, None, None]


def dask_threshold(im, cutoff_method):
    # run threshold in each chunk
    threshold_function = dask_threshold_calculator(cutoff_method)
    threshold_in_chunks = im.map_blocks(threshold_function.calculate_by_chunks, chunks=(1, 1, 1),
                                        dtype="float").compute()

    # approximate the threshold for the whole image from chunk thresholds
    valid_th = threshold_in_chunks[~np.isnan(threshold_in_chunks)]

    upper = np.percentile(valid_th, 99)
    lower = np.percentile(valid_th, 1)
    selected_th = valid_th[np.logical_and(valid_th >= lower, valid_th <= upper)]
    return np.mean(selected_th)


def segmentation(input, output, cfg):
    if cfg['small_RAM_mode'] == 1:
        # create a temporay folder to save the intermediate results
        tmp_path = "_tmp_zarr_" + str(uuid.uuid4())
        os.makedirs(tmp_path, exist_ok=True)

        fn_base = "image.zarr"

        # read the image in by delayed computing and save to ZARR
        reader = AICSImage(input)
        im = reader.get_image_dask_data("ZYX", C=0, T=0)

        raw_path = tmp_path + os.sep + fn_base
        im_dask = im.rechunk(chunks='auto')
        im_dask.to_zarr(raw_path)

        # run smoothing and save the results to another ZARR file
        # saving to ZARR file will only take disk space, not RAM
        im_dask = da.from_zarr(raw_path)
        im_smooth = da.map_overlap(edge_preserving_smoothing_3d, im_dask, dtype="int64", depth=5)

        smooth_path = tmp_path + os.sep + "smooth_" + fn_base
        im_smooth.to_zarr(smooth_path)

        # run core segmentation steps
        im_smooth = da.from_zarr(smooth_path)

        cutoff_value = ndmeasure.mean(im_smooth) + \
                       cfg["core_threshold"] * ndmeasure.standard_deviation(im_smooth)

        seg = im_smooth > cutoff_value

        threshold_path = tmp_path + os.sep + "threshold_" + fn_base
        seg.to_zarr(threshold_path)

        # run vesselness filter
        if cfg["core_vessel_1"] == 1:
            vess_func = vesselness_filter_dask(
                sigma=cfg["sigma_1"],
                gamma=cfg["gamma_1"]
            )
            vess_1 = da.map_overlap(
                vess_func.compute_vesselness,
                im_smooth,
                dtype="float",
                depth=3
            )
            vess_1_path = tmp_path + os.sep + "vess_1_" + fn_base
            vess_1.to_zarr(vess_1_path)

            # apply the cutoff
            vess_1 = da.from_zarr(vess_1_path)
            th_1 = dask_threshold(
                vess_1,
                cutoff_method=cfg["cutoff_method_1"]
            )
            seg_1 = vess_1 > th_1
            seg_1_path = tmp_path + os.sep + "seg_1_" + fn_base
            seg_1.to_zarr(seg_1_path)

        if cfg["core_vessel_2"] == 1:
            # run vesseless filter
            vess_func = vesselness_filter_dask(
                sigma=cfg["sigma_2"],
                gamma=cfg["gamma_2"]
            )
            vess_2 = da.map_overlap(
                vess_func.compute_vesselness,
                im_smooth,
                dtype="float",
                depth=3
            )
            vess_2_path = tmp_path + os.sep + "vess_2_" + fn_base
            vess_2.to_zarr(vess_2_path)

            # apply cutoff
            vess_2 = da.from_zarr(vess_2_path)
            th_2 = dask_threshold(
                vess_2,
                cutoff_method=cfg["cutoff_method_2"]
            )
            seg_2 = vess_2 > th_2
            seg_2_path = tmp_path + os.sep + "seg_2_" + fn_base
            seg_2.to_zarr(seg_2_path)

        # combine the segmentations
        latest_seg_path = None
        if cfg["core_threshold"] is not None:
            latest_seg_path = threshold_path

        if cfg["core_vessel_1"] == 1:
            # if no thresholding step (even though all current workflows use thresholding)
            if latest_seg_path is None:
                latest_seg_path = seg_1_path
            else:
                merge_path = tmp_path + os.sep + "merge_1_" + fn_base
                seg_v0 = da.from_zarr(latest_seg_path)
                seg_v1 = da.from_zarr(seg_1_path)
                seg_merge = logical_or(seg_v0, seg_v1)
                seg_merge.to_zarr(merge_path)
                latest_seg_path = merge_path

        if cfg["core_vessel_2"] == 1:
            merge_path = tmp_path + os.sep + "merge_2_" + fn_base
            seg_v0 = da.from_zarr(latest_seg_path)
            seg_v1 = da.from_zarr(seg_2_path)
            seg_merge = logical_or(seg_v0, seg_v1)
            seg_merge.to_zarr(merge_path)
            latest_seg_path = merge_path

        # post-processing
        if cfg["post_closing"] is not None:
            closing_path = tmp_path + os.sep + "closing_" + fn_base
            seg_input = da.from_zarr(latest_seg_path)
            seg_refined = ndmorph.binary_closing(
                seg_input,
                structure=cube(cfg["post_closing"]),
                border_value=1
            )
            seg_refined.to_zarr(closing_path, overwrite=True)
            latest_seg_path = closing_path

        if cfg["post_thinning"] == 1:
            thin_path = tmp_path + os.sep + "thin_" + fn_base
            seg_input = da.from_zarr(latest_seg_path)
            thin_func = partial(topology_preserving_thinning, cfg["min_thickness"], cfg["thin"])
            seg_refined = da.map_overlap(
                thin_func,
                seg_input,
                dtype="bool",
                depth=3
            )
            seg_refined.to_zarr(thin_path, overwrite=True)
            latest_seg_path = thin_path

        if cfg["post_cleaning"] is not None:
            clean_path = tmp_path + os.sep + "clean_" + fn_base
            seg_input = da.from_zarr(latest_seg_path)
            clean_func = partial(remove_small_objects, min_size=cfg["post_cleaning"])
            seg_refined = da.map_overlap(
                clean_func,
                seg_input,
                dtype="bool",
                depth=3
            )
            seg_refined.to_zarr(clean_path, overwrite=True)
            latest_seg_path = clean_path

        # generate the final segmentation as TIFF image
        final_segmentation = da.from_zarr(latest_seg_path)
        im_out = final_segmentation.compute()
        im_out = im_out.astype(np.uint8)
        im_out[im_out > 0] = 255
        OmeTiffWriter.save(im_out, output, dim_order="ZYX")

        # remove tmp folder
        rmtree(tmp_path)
    else:
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
            out = vesselness_filter(im, cfg["sigma_1"], cfg["gamma_1"], cfg["cutoff_method_1"])
            seg = np.logical_or(seg, out)

        if cfg["core_vessel_2"] == 1:
            out = vesselness_filter(im, cfg["sigma_2"], cfg["gamma_2"], cfg["cutoff_method_2"])
            seg = np.logical_or(seg, out)

        #######################
        # post-processing
        #######################
        if cfg["post_closing"] is not None:
            seg = binary_closing(seg, cube(cfg["post_closing"]))

        if cfg["post_thinning"] == 1:
            seg = topology_preserving_thinning(seg, min_thickness=cfg["min_thickness"], thin=cfg["thin"])
        if cfg["post_cleaning"] is not None:
            seg = remove_small_objects(seg, min_size=cfg["post_cleaning"])

        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255

        imsave(output, seg)


if __name__ == '__main__':
    programStart = time.time()

    parser = argparse.ArgumentParser(description='segmentation of 3D images of type .tif')

    def none_or_int(value):
        if value == 'None':
            return None
        return int(value)

    def none_or_float(value):
        if value == 'None':
            return None
        return float(value)

    parser.add_argument('-input', type=str, help='input of raw tif image')
    parser.add_argument('-prints', type=bool, default=False, help='set to True to print runtime')
    parser.add_argument('-small_RAM_mode', type=int, default=0, help='set to 1 for processing files larger than RAM')
    parser.add_argument('-smoothing', type=int, default=1, help='set to 1 for smoothing, 0 for no smoothing')
    parser.add_argument('-core_threshold', type=none_or_float, default=3.0)
    parser.add_argument('-core_vessel_1', type=int, default=1, help='set to 1 for first vesselness filter, '
                                                                    '0 for no vesselness filter')
    parser.add_argument('-gamma_1', type=int, default=5)
    parser.add_argument('-sigma_1', type=float, default=1.0)
    parser.add_argument('-cutoff_method_1', type=str, default='threshold_li')
    parser.add_argument('-core_vessel_2', type=int, default=1, help='set to 1 for second vesselness filter, '
                                                                    '0 for no vesselness filter')
    parser.add_argument('-gamma_2', type=int, default=5)
    parser.add_argument('-sigma_2', type=float, default=2.0)
    parser.add_argument('-cutoff_method_2', type=str, default='threshold_li')
    parser.add_argument('-post_closing', type=none_or_int, default=5)
    parser.add_argument('-post_thinning', type=int, default=0, help='set to 1 for post-processing, '
                                                                    '0 for no post-processing')
    parser.add_argument('-min_thickness', type=int, default=None)
    parser.add_argument('-thin', type=int, default=None)
    parser.add_argument('-post_cleaning', type=none_or_int, default=100)
    args = parser.parse_args()

    config = {
        "small_RAM_mode": args.small_RAM_mode,
        "smoothing": args.smoothing,
        "core_threshold": args.core_threshold,
        "core_vessel_1": args.core_vessel_1,
        "gamma_1": args.gamma_1,
        "sigma_1": args.sigma_1,
        "cutoff_method_1": args.cutoff_method_1,
        "core_vessel_2": args.core_vessel_2,
        "gamma_2": args.gamma_2,
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
