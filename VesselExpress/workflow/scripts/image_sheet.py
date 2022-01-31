import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile


def img_is_color(img):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10),
                    title_fontsize=30, result_dir=None):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        title = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)
        list_axes[i].axis('off')

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    #_ = plt.show()

    if result_dir != None:
        plt.savefig(os.path.join(result_dir, 'image_sheet.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates an image sheet of the segmentation results of VesselExpress')
    parser.add_argument('-dir', type=str, help='output directroy of VesselExpress')
    args = parser.parse_args()

    # All subdirectories in the current directory, not recursive.
    img_dirs = [f for f in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, f)) and not f.startswith('.')]

    fl_names = []
    imgs = []

    # loop through every image
    for i, img in enumerate(img_dirs):
        # read in raw and binary image
        img_dir = os.path.join(args.dir, img)
        raw_img = tifffile.imread(os.path.join(img_dir, img + '.tiff'))
        bin_img = tifffile.imread(os.path.join(img_dir, 'Binary_' + img + '.tiff'))

        # adjust the contrast of the raw image
        image = raw_img
        upper_thr = np.percentile(image, 99)
        lower_thr = np.percentile(image, 1)
        image[image > upper_thr] = upper_thr
        image[image < lower_thr] = lower_thr
        image = (image - lower_thr) / (upper_thr - lower_thr)

        # take middle slice and append to list
        slice = int(image.shape[0] / 2)
        fl_names.append(img)
        fl_names.append('Binary_' + img)
        imgs.append(image[slice, :, :])
        imgs.append(bin_img[slice, :, :])

    show_image_list(imgs, fl_names, num_cols=8, title_fontsize=8, figsize=(15, 10), result_dir=args.dir)
