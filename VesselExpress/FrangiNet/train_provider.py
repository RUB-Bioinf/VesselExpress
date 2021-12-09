import numpy as np
import tensorflow as tf
import os
import json
from random import shuffle
from skimage import io
from skimage.transform import resize
# import copy

from FrangiNet.metrics import show_array_info


class DataProvider:
    def __init__(self, config_file):
        """
            Read in parameters from the json-file and call
            the function to initialize the training images.
            ----------
            config_file : string, path of the json-file
            containing the parameters for the training.
        """
        f = open(config_file, 'r')
        data = json.load(f)

        self.mode = data['frangi_net']['mode']
        self.gpus = data['frangi_net']['gpus']

        self.path_image = data['frangi_net']['path_image']
        self.path_valid_image = data['frangi_net']['path_valid_image']
        self.path_binary = data['frangi_net']['path_binary']
        self.path_valid_binary = data['frangi_net']['path_valid_binary']
        self.path_save = data['frangi_net']['path_save']
        self.path_model = data['frangi_net']['path_model']

        self.use_mask = data['frangi_net']['use_mask']
        self.path_mask = data['frangi_net']['path_mask']
        self.path_valid_mask = data['frangi_net']['path_valid_mask']

        self.use_weight = data['frangi_net']['use_weight']
        self.path_weight = data['frangi_net']['path_weight']
        self.path_valid_weight = data['frangi_net']['path_valid_weight']

        self.step_summary = data['frangi_net']['epoch_summary']
        self.step_save = data['frangi_net']['epoch_save']
        self.step_overall = data['frangi_net']['epoch_overall']
        self.size_batch = data['frangi_net']['size_batch']
        self.rotate_images = data['frangi_net']['rotate_images']
        self.normalization = data['frangi_net']['normalize_images']
        self.threshold_metric = data['frangi_net']['threshold_metric']

        self.is_restore = data['frangi_net']['is_restore']
        self.step_restore = data['frangi_net']['step_restore']

        self.restrict_roc = data['frangi_net']['restrict_roc']
        self.restrict_amount = data['frangi_net']['restrict_image_amount']

        self.create_roc = data['frangi_net']['create_roc']
        self.create_prc = data['frangi_net']['create_prc']

        self.early_stopping = data['frangi_net']['early_stopping']
        self.delta_loss = data['frangi_net']['delta_loss']

        self.image_big = (0, 0)
        self.image_small = (0, 0)  # reduced from image_big by factor 0.25

        self.num_channel = 1
        self.num_class = 2
        self.reg_sim = 0

        self.rate_learning = 1e-4
        self.rate_decay = 0.9
        self.step_decay = 2000
        self.step_global = tf.Variable(0., name='global_step')

        # ###
        # Variables for Training
        self.image = None
        self.binary_image = None
        self.weight_image = None
        self.mask_image = None
        self.number_slices = 0  # amount of individual images
        self.train_sets = None  # contains all files which belong to the training set
        self.index = 0
        self.current_images = None  # stores the images of the current batch in training
        self.steps_per_epoch = None

        # ###
        # Variables for Validation

        self.valid_image = None
        self.valid_binary_image = None
        self.valid_weight_image = None
        self.valid_mask_image = None
        self.valid_number_slices = 0
        self.valid_sets = None  # contains all files which belong to the validation set
        self.index_valid = 0
        self.current_images_valid = None
        self.steps_per_validation = None

        # load all images into memory and prepare them for training/validation
        self.init_image()

        # needed tensors for the neural network
        self.x = tf.placeholder(tf.float32, [None, self.image_small[0], self.image_small[1], self.num_channel],
                                name='input_x')
        self.y = tf.placeholder(tf.float32, [None, self.image_big[0], self.image_big[1], self.num_class],
                                name='input_y')
        self.m = tf.placeholder(tf.bool, [None, self.image_big[0], self.image_big[1], self.num_class],
                                name='input_m')
        self.w = tf.placeholder(tf.float32, [None, self.image_big[0], self.image_big[1], self.num_class],
                                name='input_w')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='input_is_train')

    # ##########################
    # 2D & 3D ROTATION
    ############################

    def image_setting(self, image_files, path_image, path_binary, path_mask, path_weight, text):
        temp_bin = io.imread(path_binary + image_files[0])
        binary_factor = np.max(temp_bin)
        image = None
        mask_image = None
        weight_image = None

        for file in image_files:
            if file.endswith('.tif') | file.endswith('.tiff') | file.endswith('.png') | file.endswith('.jpg'):
                print("File Loading " + text + ": " + file)
                im = io.imread(path_image + file)
                im_bin = io.imread(path_binary + file)

                if self.normalization is True:
                    maximum = np.max(im)
                    minimum = np.min(im)
                    im = (im - minimum) / (maximum - minimum)

                # images need to be reduced by factor 4 and fit into value range -1, 1
                net_image = np.zeros(shape=(im.shape[0], self.image_small[0], self.image_small[1]), dtype=np.float32)
                net_label = np.zeros(shape=(im_bin.shape[0], self.image_big[0], self.image_big[1]), dtype=np.float32)
                if self.use_mask is True:
                    im_mask = io.imread(path_mask + file)
                    net_mask = np.zeros(shape=(im_mask.shape[0], self.image_big[0], self.image_big[1]), dtype=np.bool)
                if self.use_weight is True:
                    im_weight = io.imread(path_weight + file)
                    net_weight = np.zeros(shape=(im_weight.shape[0], self.image_big[0], self.image_big[1]),
                                          dtype=np.float32)

                for i in range(0, im.shape[0]):
                    net_image[i] = resize(im[i], (self.image_small[0], self.image_small[1]), anti_aliasing=False)
                    net_image[i] = (net_image[i] * 2) - 1

                    net_label[i] = im_bin[i, :self.image_big[0], :self.image_big[1]]
                    net_label[i] = net_label[i] / binary_factor

                    if self.use_mask is True:
                        net_mask[i] = im_mask[i, :self.image_big[0], :self.image_big[1]]
                    if self.use_weight is True:
                        net_weight[i] = im_weight[i, :self.image_big[0], :self.image_big[1]]

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
                    if self.use_weight is True:
                        net_weight_y = np.zeros(shape=(self.image_big[0], self.image_big[0], self.image_big[1]),
                                                dtype=np.float32)

                    # rotation
                    im_y = np.rot90(im[0:self.image_big[0], :, :], axes=(0, 1))
                    im_bin_y = np.rot90(im_bin[0:self.image_big[0], :, :], axes=(0, 1))
                    if self.use_mask is True:
                        im_mask_y = np.rot90(im_mask[0:self.image_big[0], :, :], axes=(0, 1))
                    if self.use_weight is True:
                        im_weight_y = np.rot90(im_weight[0:self.image_big[0], :, :], axes=(0, 1))

                    # fill in the values
                    for i in range(0, im_y.shape[0]):
                        net_image_y[i] = resize(im_y[i], (self.image_small[0], self.image_small[1]),
                                                anti_aliasing=False)
                        net_image_y[i] = (net_image_y[i] * 2) - 1

                        net_label_y[i] = im_bin_y[i] / binary_factor

                        if self.use_mask is True:
                            net_mask_y[i] = im_mask_y[i, :self.image_big[0], :self.image_big[1]]
                        if self.use_weight is True:
                            net_weight_y[i] = im_weight_y[i, :self.image_big[0], :self.image_big[1]]

                    # ## ROTATE Z DIMENSION
                    # init images
                    net_image_z = np.zeros(shape=(self.image_big[0], self.image_small[0], self.image_small[1]),
                                           dtype=np.float32)
                    net_label_z = np.zeros(shape=(self.image_big[0], self.image_big[0], self.image_big[1]),
                                           dtype=np.float32)
                    if self.use_mask is True:
                        net_mask_z = np.zeros(shape=(self.image_big[0], self.image_big[0], self.image_big[1]),
                                              dtype=np.bool)
                    if self.use_weight is True:
                        net_weight_z = np.zeros(shape=(self.image_big[0], self.image_big[0], self.image_big[1]),
                                                dtype=np.float32)

                    # rotation
                    im_z = np.rot90(im[0:self.image_big[0], :, :], axes=(0, 2))
                    im_bin_z = np.rot90(im_bin[0:self.image_big[0], :, :], axes=(0, 2))
                    if self.use_mask is True:
                        im_mask_z = np.rot90(im_mask[0:self.image_big[0], :, :], axes=(0, 2))
                    if self.use_weight is True:
                        im_weight_z = np.rot90(im_weight[0:self.image_big[0], :, :], axes=(0, 2))

                    # fill in the values
                    for i in range(0, im_z.shape[0]):
                        net_image_z[i] = resize(im_z[i], (self.image_small[0], self.image_small[1]),
                                                anti_aliasing=False)
                        net_image_z[i] = (net_image_z[i] * 2) - 1

                        net_label_z[i] = im_bin_z[i]
                        net_label_z[i] = net_label_z[i] / binary_factor

                        if self.use_mask is True:
                            net_mask_z[i] = im_mask_z[i, :self.image_big[0], :self.image_big[1]]
                        if self.use_weight is True:
                            net_weight_z[i] = im_weight_z[i, :self.image_big[0], :self.image_big[1]]
                            net_weight_z[i] = net_weight_z[i] / 255

                    temp_net_image = np.concatenate((net_image, net_image_y), axis=0)
                    temp_net_image = np.concatenate((temp_net_image, net_image_z), axis=0)

                    temp_net_label = np.concatenate((net_label, net_label_y), axis=0)
                    temp_net_label = np.concatenate((temp_net_label, net_label_z), axis=0)

                    if self.use_mask is True:
                        temp_net_mask = np.concatenate((net_mask, net_mask_y), axis=0)
                        temp_net_mask = np.concatenate((temp_net_mask, net_mask_z), axis=0)
                    if self.use_weight is True:
                        temp_net_weight = np.concatenate((net_weight, net_weight_y), axis=0)
                        temp_net_weight = np.concatenate((temp_net_weight, net_weight_z), axis=0)

                    if image is not None:
                        image = np.concatenate((image, temp_net_image), axis=0)
                        binary_image = np.concatenate((binary_image, temp_net_label), axis=0)
                        if self.use_mask is True:
                            mask_image = np.concatenate((mask_image, temp_net_mask), axis=0)
                        if self.use_weight is True:
                            weight_image = np.concatenate((weight_image, temp_net_weight), axis=0)
                    else:  # very first image needs to be put into the variable, afterwards images are concatenated
                        image = temp_net_image
                        binary_image = temp_net_label
                        if self.use_mask is True:
                            mask_image = temp_net_mask
                        if self.use_weight is True:
                            weight_image = temp_net_weight

                else:  # rotation is not activated
                    if image is not None:
                        image = np.concatenate((image, net_image), axis=0)
                        binary_image = np.concatenate((binary_image, net_label), axis=0)
                        if self.use_mask is True:
                            mask_image = np.concatenate((mask_image, net_mask), axis=0)
                        if self.use_weight is True:
                            weight_image = np.concatenate((weight_image, net_weight), axis=0)
                    else:
                        image = net_image
                        binary_image = net_label
                        if self.use_mask is True:
                            mask_image = net_mask
                        if self.use_weight is True:
                            weight_image = net_weight

        return image, binary_image, mask_image, weight_image

    def init_image(self):
        """
            Puts all the files as into memory. Image stacks are
            stored into one variable containing all individual images.
            If rotation is activated, the stacks are rotated for
            adding the y and z dimension.
            Training and validation images are stored in separate variables.
            Same goes for their respective label images.
            The images are prepared in that they are reduced by factor 4
            and net_image to value range -1, 1.
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

        # ###
        # TRAINING AND VALIDATION IMAGE

        self.image, self.binary_image, self.mask_image, self.weight_image = self.image_setting(image_files,
            self.path_image, self.path_binary, self.path_mask, self.path_weight, "Train")

        image_files = os.listdir(self.path_valid_image)
        self.valid_image, self.valid_binary_image, self.valid_mask_image, self.valid_weight_image = self.image_setting(
            image_files, self.path_valid_image, self.path_valid_binary, self.path_valid_mask, self.path_valid_weight,
            "Valid")

        # net_labelulate from epochs to steps (batches)
        self.number_slices = self.image.shape[0]
        self.steps_per_epoch = int(self.number_slices / self.size_batch) + (self.number_slices % self.size_batch > 0)
        self.valid_number_slices = self.valid_image.shape[0]
        self.steps_per_validation = int(self.valid_number_slices / self.size_batch) + (
                self.valid_number_slices % self.size_batch > 0)
        self.step_summary = self.step_summary * self.steps_per_epoch
        self.step_save = self.step_save * self.steps_per_epoch
        self.step_overall = self.step_overall * self.steps_per_epoch

        # list to shuffle images for different batches in epochs
        setlist = np.arange(0, self.valid_number_slices)
        np.random.shuffle(setlist)
        self.valid_sets = setlist

        # list to shuffle images for different batches in epochs
        setlist = np.arange(0, self.number_slices)
        np.random.shuffle(setlist)
        self.train_sets = setlist

        show_array_info("Image", self.image)
        show_array_info("Binary", self.binary_image)
        if self.use_mask is True:
            show_array_info("Mask", self.mask_image)
        if self.use_weight is True:
            show_array_info("Weight", self.weight_image)

        show_array_info("Valid Image", self.valid_image)
        show_array_info("Valid Binary", self.valid_binary_image)
        if self.use_mask is True:
            show_array_info("Valid Mask", self.valid_mask_image)
        if self.use_weight is True:
            show_array_info("Valid Weight", self.valid_weight_image)

    def provide_valid(self):
        """
            Prepares images of current batch which are given
            to the neural network for validation.
            -------------
            x : array, contains the images
            y : array, contains the corresponding labels
            m : array, contains the corresponding mask
            w : array, contains the corresponding weighted mask
        """
        self.current_images_valid = []  # save current images in batch for metrics
        for i in range(self.size_batch):
            self.current_images_valid.append(self.valid_sets[self.index_valid])

            if self.index_valid < len(self.valid_sets) - 1:
                self.index_valid += 1
            else:  # last batch might not have enough images, leave early
                shuffle(self.valid_sets)
                self.index_valid = 0
                break

        x = np.zeros(shape=(len(self.current_images_valid), self.image_small[0], self.image_small[1], self.num_channel),
                     dtype=np.float32)
        y = np.zeros(shape=(len(self.current_images_valid), self.image_big[0], self.image_big[1], self.num_class),
                     dtype=np.float32)
        m = np.ones(shape=(len(self.current_images_valid), self.image_big[0], self.image_big[1], self.num_class),
                    dtype=np.bool)
        w = np.ones(shape=(len(self.current_images_valid), self.image_big[0], self.image_big[1], self.num_class),
                    dtype=np.float32)

        # fill images into the batch
        for i in range(len(self.current_images_valid)):
            x[i, :, :, 0] = self.valid_image[self.current_images_valid[i]]
            y[i, :, :, 1] = self.valid_binary_image[self.current_images_valid[i]]
            y[i, :, :, 0: 1] = 1 - y[i, :, :, 1:]
            if self.use_mask is True:
                m[i, :, :, 0] = self.valid_mask_image[self.current_images_valid[i]]
            m[i, :, :, 1:] = m[i, :, :, 0: 1]
            if self.use_weight is True:
                w[i, :, :, 0] = self.valid_weight_image[self.current_images_valid[i]]
            w[i, :, :, 1:] = w[i, :, :, 0: 1]

        return x, y, m, w

    def provide_train(self):
        """
            Prepares images of current batch which are given
            to the neural network for validation.
            -------------
            x : array, contains the images
            y : array, contains the corresponding labels
            m : array, contains the corresponding mask
            w : array, contains the corresponding weighted mask
        """
        self.current_images = []  # save current images in batch for metrics
        for i in range(self.size_batch):
            self.current_images.append(self.train_sets[self.index])

            if self.index < len(self.train_sets) - 1:
                self.index += 1
            else:  # last batch might not have enough images, leave early
                shuffle(self.train_sets)
                self.index = 0
                break

        x = np.zeros(shape=(len(self.current_images), self.image_small[0], self.image_small[1], self.num_channel),
                     dtype=np.float32)
        y = np.zeros(shape=(len(self.current_images), self.image_big[0], self.image_big[1], self.num_class),
                     dtype=np.float32)
        m = np.ones(shape=(len(self.current_images), self.image_big[0], self.image_big[1], self.num_class),
                    dtype=np.bool)
        w = np.ones(shape=(len(self.current_images), self.image_big[0], self.image_big[1], self.num_class),
                    dtype=np.float32)

        # fill images into the batch
        for i in range(len(self.current_images)):
            x[i, :, :, 0] = self.image[self.current_images[i]]
            y[i, :, :, 1] = self.binary_image[self.current_images[i]]
            y[i, :, :, 0: 1] = 1 - y[i, :, :, 1:]
            if self.use_mask is True:
                m[i, :, :, 0] = self.mask_image[self.current_images[i]]
            m[i, :, :, 1:] = m[i, :, :, 0: 1]
            if self.use_weight is True:
                w[i, :, :, 0] = self.weight_image[self.current_images[i]]
            w[i, :, :, 1:] = w[i, :, :, 0: 1]

        return x, y, m, w
