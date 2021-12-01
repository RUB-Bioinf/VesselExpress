import os

import tensorflow as tf
import numpy as np
import math
from skimage import io
from skimage.transform import rescale, resize
import json


class DataProvider:
    def __init__(self, config_file):
        f = open(config_file, 'r')
        data = json.load(f)

        self.mode = data['frangi_net']['mode']
        self.gpus = data['frangi_net']['gpus']

        self.path_image = data['frangi_net']['path_image']
        self.path_binary = data['frangi_net']['path_binary']
        self.path_save = data['frangi_net']['path_save']
        self.path_model = data['frangi_net']['path_model']

        self.use_mask = data['frangi_net']['use_mask']
        self.path_mask = data['frangi_net']['path_mask']

        self.size_batch = data['frangi_net']['size_batch']
        self.rotate_images = data['frangi_net']['rotate_images']
        self.normalization = data['frangi_net']['normalize_images']

        self.threshold_metric = data['frangi_net']['threshold_metric']

        self.create_roc = data['frangi_net']['create_roc']
        self.create_prc = data['frangi_net']['create_prc']

        self.image_big = (0, 0)
        self.image_small = (0, 0)  # reduced from image_big by factor 0.25

        self.result_image = None

        self.num_channel = 1
        self.num_class = 2

        # ###
        # Variables for Training
        self.image = None
        self.binary_image = None
        self.mask_image = None
        self.index = 0
        self.current_images = None  # stores the images of the current batch in training

        # load all images into memory and prepare them for training/validation
        self.init_image()

        # necessary tensors for the neural network
        self.x = tf.placeholder(tf.float32, [None, self.image_small[0], self.image_small[1], self.num_channel],
                                name='input_x')
        self.m = tf.placeholder(tf.bool, [None, self.image_big[0], self.image_big[1], self.num_class],
                                name='input_m')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='input_is_train')

    def init_image(self):
        """
            Puts all the files into memory. Image stacks are
            stored into one variable containing all individual images.
            If rotation is activated, the stacks are rotated for
            adding the y and z dimension.
            The images are prepared in that they are reduced by factor 4
            and mapped to value range -1, 1.
        """
        image_files = os.listdir(self.path_image)
        temp = io.imread(self.path_image + image_files[0])

        if self.rotate_images is True:
            min_dim = min(temp.shape[0], temp.shape[1], temp.shape[2])
            if min_dim % 4 != 0:
                size = min_dim - min_dim % 4
            else:
                size = min_dim
            self.image_big = (size, size)
            self.image_small = (int(self.image_big[0] / 4), int(self.image_big[0] / 4))
        else:
            if temp.shape[2] % 4 != 0:
                size_x = temp.shape[2] - temp.shape[2] % 4
            else:
                size_x = temp.shape[2]
            if temp.shape[1] % 4 != 0:
                size_y = temp.shape[1] - temp.shape[1] % 4
            else:
                size_y = temp.shape[1]
            self.image_big = (size_x, size_y)
            self.image_small = (int(self.image_big[0] / 4), int(self.image_big[0] / 4))

        temp_bin = io.imread(self.path_binary + image_files[0])
        binary_factor = np.max(temp_bin)
        image = None
        mask_image = None

        for file in image_files:
            if file.endswith('.tif') | file.endswith('.tiff') | file.endswith('.png') | file.endswith('.jpg'):
                print("File Loading: " + file)
                im = io.imread(self.path_image + file)
                im_bin = io.imread(self.path_binary + file)

                if self.normalization is True:
                    maximum = np.max(im)
                    minimum = np.min(im)
                    im = (im - minimum) / (maximum - minimum)

                # images need to be reduced by factor 4 and fit into value range -1, 1
                net_image = np.zeros(shape=(im.shape[0], self.image_small[0], self.image_small[1]), dtype=np.float32)
                net_label = np.zeros(shape=(im_bin.shape[0], self.image_big[0], self.image_big[1]), dtype=np.float32)
                if self.use_mask is True:
                    im_mask = io.imread(self.path_mask + file)
                    net_mask = np.zeros(shape=(im_mask.shape[0], self.image_big[0], self.image_big[1]), dtype=np.bool)

                for i in range(0, im.shape[0]):
                    net_image[i] = resize(im[i], (self.image_small[0], self.image_small[1]), anti_aliasing=False)
                    net_image[i] = (net_image[i] * 2) - 1

                    net_label[i] = im_bin[i, :self.image_big[0], :self.image_big[1]]
                    net_label[i] = net_label[i] / binary_factor

                    if self.use_mask is True:
                        net_mask[i] = im_mask[i, :self.image_big[0], :self.image_big[1]]

                if self.rotate_images is True:
                    # ## ROTATE Y DIMENSION
                    # init images
                    net_image_y = np.zeros(shape=(self.image_big[0], self.image_small[0], self.image_small[1]),
                                           dtype=np.float32)
                    net_label_y = np.zeros(shape=(self.image_big[0], self.image_big[0], self.image_big[1]),
                                           dtype=np.float32)
                    if self.use_mask is True:
                        net_mask_y = np.zeros(shape=(self.image_big[0], self.image_big[0], self.image_big[1]),
                                              dtype=np.bool)

                    # rotation
                    im_y = np.rot90(im[0:self.image_big[0], :, :], axes=(0, 1))
                    im_bin_y = np.rot90(im_bin[0:self.image_big[0], :, :], axes=(0, 1))
                    if self.use_mask is True:
                        im_mask_y = np.rot90(im_mask[0:self.image_big[0], :, :], axes=(0, 1))

                    # fill in the values
                    for i in range(0, im_y.shape[0]):
                        net_image_y[i] = resize(im_y[i], (self.image_small[0], self.image_small[1]),
                                                anti_aliasing=False)
                        net_image_y[i] = (net_image_y[i] * 2) - 1

                        net_label_y[i] = im_bin_y[i] / binary_factor

                        if self.use_mask is True:
                            net_mask_y[i] = im_mask_y[i, :self.image_big[0], :self.image_big[1]]

                    # ## ROTATE Z DIMENSION
                    # init images
                    net_image_z = np.zeros(shape=(self.image_big[0], self.image_small[0], self.image_small[1]),
                                           dtype=np.float32)
                    net_label_z = np.zeros(shape=(self.image_big[0], self.image_big[0], self.image_big[1]),
                                           dtype=np.float32)
                    if self.use_mask is True:
                        net_mask_z = np.zeros(shape=(self.image_big[0], self.image_big[0], self.image_big[1]),
                                              dtype=np.bool)

                    # rotation
                    im_z = np.rot90(im[0:self.image_big[0], :, :], axes=(0, 2))
                    im_bin_z = np.rot90(im_bin[0:self.image_big[0], :, :], axes=(0, 2))
                    if self.use_mask is True:
                        im_mask_z = np.rot90(im_mask[0:self.image_big[0], :, :], axes=(0, 2))

                    # fill in the values
                    for i in range(0, im_z.shape[0]):
                        net_image_z[i] = resize(im_z[i], (self.image_small[0], self.image_small[1]),
                                                anti_aliasing=False)
                        net_image_z[i] = (net_image_z[i] * 2) - 1

                        net_label_z[i] = im_bin_z[i]
                        net_label_z[i] = net_label_z[i] / binary_factor

                        if self.use_mask is True:
                            net_mask_z[i] = im_mask_z[i, :self.image_big[0], :self.image_big[1]]

                    temp_net_image = np.concatenate((net_image, net_image_y), axis=0)
                    temp_net_image = np.concatenate((temp_net_image, net_image_z), axis=0)

                    temp_net_label = np.concatenate((net_label, net_label_y), axis=0)
                    temp_net_label = np.concatenate((temp_net_label, net_label_z), axis=0)

                    if self.use_mask is True:
                        temp_net_mask = np.concatenate((net_mask, net_mask_y), axis=0)
                        temp_net_mask = np.concatenate((temp_net_mask, net_mask_z), axis=0)

                    if self.image is not None:
                        self.image = np.concatenate((self.image, temp_net_image), axis=0)
                        self.binary_image = np.concatenate((self.binary_image, temp_net_label), axis=0)
                        if self.use_mask is True:
                            self.mask_image = np.concatenate((self.mask_image, temp_net_mask), axis=0)
                    else:  # very first image needs to be put into the variable, afterwards images are concatenated
                        self.image = temp_net_image
                        self.binary_image = temp_net_label
                        if self.use_mask is True:
                            self.mask_image = temp_net_mask

                else:  # rotation is not activated
                    if self.image is not None:
                        self.image = np.concatenate((self.image, net_image), axis=0)
                        self.binary_image = np.concatenate((self.binary_image, net_label), axis=0)
                        if self.use_mask is True:
                            self.mask_image = np.concatenate((self.mask_image, net_mask), axis=0)
                    else:
                        self.image = net_image
                        self.binary_image = net_label
                        if self.use_mask is True:
                            self.mask_image = net_mask

        print(self.image.shape)
        print(self.binary_image.shape)

        self.result_image = np.zeros(shape=(self.image.shape[0], self.image_big[0], self.image_big[1]), dtype=np.float32)

    def provide_test(self):
        """
            Prepares image and its mask that are then given
            to the neural network to be processed.
            ------------
            x : array, contains the images
            m : array, contains the corresponding mask
        """
        x = np.zeros(shape=(1, self.image_small[0], self.image_small[1], self.num_channel), dtype=np.float32)
        x[:, :, :, 0] = self.image[self.index]

        m = np.ones(shape=(1, self.image_big[0], self.image_big[1], self.num_class), dtype=np.bool)
        m[:, :, :, 1:] = m[:, :, :, 0:1]
        self.index = self.index + 1

        return x, m
