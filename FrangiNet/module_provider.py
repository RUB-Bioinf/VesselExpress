import tensorflow as tf
import numpy as np
import math
from skimage import io
from skimage.transform import rescale, resize


class DataProvider:
    def __init__(self, i, norm, mode, batch_size):
        """
            Set and calculate parameters for the image to be
            processed on the neural network
            ----------
            i : string, path of the image to be segmented
            norm : boolean, normalize image before processing
            mode_img: string, "3DCube" for splitting the image in
            cubes to be processed in all three dimensions, "OneCube"
            to use a single cube to be processed in all three
            dimensions, "Default" handles the image as single
            2D images that are stacked after being processed
        """
        self.num_class = 2
        self.num_channel = 1
        self.mode = mode
        self.normalization = norm

        self.index = 0
        self.small_size = (0, 0)  # image needs to be rescaled by factor 0.25
        self.big_size = (0, 0)  # original dimensions of the image
        self.factor4 = 0  # used if image dimension is not a multiple of 4
        # TODO: take into consideration that dimensions may need different factors
        # only needed for the original approach
        self.result_image = None  #

        if self.mode == "3DCube":
            self.cube_size = 248
            # TODO: may be set by the user, should be a multiple of 4
            self.number_cubes = 0
            self.positions = []  # decide where each cube has to be placed after being processed
            self.image_x = None
            self.image_y = None
            self.image_z = None
            self.init_test_image_3dcube(i)
        elif self.mode == "OneCube" or self.mode == "OneCubeBatch":
            self.image_x = None
            self.image_y = None
            self.image_z = None
            self.image_mask_x = None
            self.image_mask_y = None
            self.image_mask_z = None
            self.batch_size = batch_size
            self.cur_size = 0  # needed if size of image not a multiple of batch_size
            self.dim_xyz = (0, 0, 0)  # dimension of the original image
            self.init_test_image_onecube(i)
        else:
            self.image = None
            self.stack_size = None
            self.init_test_image(i)

        # necessary tensors for the neural network
        self.x = tf.placeholder(tf.float32, [None, self.small_size[0], self.small_size[1], self.num_channel],
                                name='input_x')
        self.m = tf.placeholder(tf.bool, [None, self.big_size[0], self.big_size[1], self.num_class],
                                name='input_m')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='input_is_train')

    def init_test_image_onecube(self, image_path):
        """
            Utilizes the technique of processing 3D image as a single
            cube. The image is stored in three different variables, each
            representing one dimension (x, y, z).
            The cube has the size of the highest dimension.
            For each dimension the appropriate mask is calculated.
            ----------
            image_patch : string, path of the image to be segmented
        """
        img = io.imread(image_path)
        self.dim_xyz = (img.shape[2], img.shape[1], img.shape[0])

        if self.normalization is True:
            print("Images are normalized")
            maximum = np.max(img)
            minimum = np.min(img)
            img = (img - minimum) / (maximum - minimum)

        # get the highest dimension and check of it is a multiple of 4, else add factor
        max_dim = max(self.dim_xyz[0], self.dim_xyz[1], self.dim_xyz[2])
        if max_dim % 4 != 0:
            self.factor4 = 4 - max_dim % 4
        self.big_size = (max_dim + self.factor4, max_dim + self.factor4)
        self.small_size = (int(self.big_size[0] / 4), int(self.big_size[0] / 4))

        # images need to be resized by factor 0.25
        self.image_x = np.zeros(shape=(self.dim_xyz[2], self.small_size[0], self.small_size[0]), dtype=np.float32)
        self.image_y = np.zeros(shape=(self.dim_xyz[1], self.small_size[0], self.small_size[0]), dtype=np.float32)
        self.image_z = np.zeros(shape=(self.dim_xyz[0], self.small_size[0], self.small_size[0]), dtype=np.float32)

        # ###
        # Fill in x, y and z image with the respective dimension
        # ###

        for i in range(0, self.dim_xyz[2]):
            self.image_x[i, :int(self.dim_xyz[1]/4), :int(self.dim_xyz[0]/4)] = resize(img[i, :, :],
                                (int(self.dim_xyz[1]/4), int(self.dim_xyz[0]/4)), anti_aliasing=False)
            self.image_x[i] = (self.image_x[i] * 2) - 1  # values need to be in range -1, 1

        for i in range(0, self.dim_xyz[1]):
            self.image_y[i, :int(self.dim_xyz[2]/4), :int(self.dim_xyz[0]/4)] = resize(img[:, i, :],
                                (int(self.dim_xyz[2]/4), int(self.dim_xyz[0]/4)), anti_aliasing=False)
            self.image_y[i] = (self.image_y[i] * 2) - 1

        for i in range(0, self.dim_xyz[0]):
            self.image_z[i, :int(self.dim_xyz[2]/4), :int(self.dim_xyz[1]/4)] = resize(img[:, :, i],
                                (int(self.dim_xyz[2]/4), int(self.dim_xyz[1]/4)), anti_aliasing=False)
            self.image_z[i] = (self.image_z[i] * 2) - 1

        # ###
        # Create appropriate mask for each dimension image
        # ###

        self.image_mask_x = np.zeros(shape=(self.dim_xyz[2], self.big_size[0], self.big_size[0]), dtype=np.bool)
        temp_mask = np.ones(shape=(self.dim_xyz[2], self.dim_xyz[1], self.dim_xyz[0]), dtype=np.bool)
        self.image_mask_x[:, :self.dim_xyz[1], :self.dim_xyz[0]] = temp_mask

        self.image_mask_y = np.zeros(shape=(self.dim_xyz[1], self.big_size[0], self.big_size[0]), dtype=np.bool)
        temp_mask = np.ones(shape=(self.dim_xyz[1], self.dim_xyz[2], self.dim_xyz[0]), dtype=np.bool)
        self.image_mask_y[:, :self.dim_xyz[2], :self.dim_xyz[0]] = temp_mask

        self.image_mask_z = np.zeros(shape=(self.dim_xyz[0], self.big_size[0], self.big_size[0]), dtype=np.bool)
        temp_mask = np.ones(shape=(self.dim_xyz[0], self.dim_xyz[2], self.dim_xyz[1]), dtype=np.bool)
        self.image_mask_z[:, :self.dim_xyz[2], :self.dim_xyz[1]] = temp_mask

    def provide_test_onecube(self, dim):
        """
            Prepares image and its mask that are then given
            to the neural network to be processed.
            ----------
            dim : string, dimension that is being processed
                either x, y or z
            ----------
            x : array, contains the images
            m : array, contains the corresponding mask
        """
        if dim == "x":
            image = self.image_x
            mask = self.image_mask_x
            size = self.dim_xyz[2]
        elif dim == "y":
            image = self.image_y
            mask = self.image_mask_y
            size = self.dim_xyz[1]
        else:
            image = self.image_z
            mask = self.image_mask_z
            size = self.dim_xyz[0]

        x = np.zeros(shape=(size, self.small_size[0], self.small_size[1], self.num_channel), dtype=np.float32)
        x[:, :, :, 0] = image

        m = np.ones(shape=(size, self.big_size[0], self.big_size[0], self.num_class), dtype=np.bool)
        m[:size, :, :, 0] = mask[:size, :, :]
        m[:size, :, :, 1:] = m[:, :, :, 0:1]

        return x, m

    def provide_test_onecube_batch(self, dim):
        """
            Prepares image and its mask that are then given
            to the neural network to be processed in the
            size of a batch.
            ----------
            dim : string, dimension that is being processed
                either x, y or z
            ----------
            x : array, contains the images
            m : array, contains the corresponding mask
        """
        if dim == "x":
            image = self.image_x
            mask = self.image_mask_x
            size = self.dim_xyz[2]
        elif dim == "y":
            image = self.image_y
            mask = self.image_mask_y
            size = self.dim_xyz[1]
        else:
            image = self.image_z
            mask = self.image_mask_z
            size = self.dim_xyz[0]

        if self.index + self.batch_size > size:
            self.cur_size = self.batch_size - (self.index + self.batch_size - size)
        else:
            self.cur_size = self.batch_size

        x = np.zeros(shape=(self.cur_size, self.small_size[0], self.small_size[1], self.num_channel), dtype=np.float32)
        m = np.ones(shape=(self.cur_size, self.big_size[0], self.big_size[0], self.num_class), dtype=np.bool)
        for i in range(self.cur_size):
            x[i, :, :, 0] = image[self.index]
            m[i, :, :, 0] = mask[i, :, :]
            m[i, :, :, 1:] = m[i, :, :, 0:1]
            self.index += 1

        return x, m

    def init_test_image_3dcube(self, image_path):
        """
            Utilizes the technique of processing 3D image as a collection
            of equal sized cubes. Each cube is then processed in all three
            dimensions by rotating it. At the end, all cubes are added to
            the result image.
            Border handling: The cubes at the border are repositioned by
            the amount of pixels they else would be out of boundary.
            ----------
            image_patch : string, path of the image to be segmented
        """
        img = io.imread(image_path)

        if self.normalization is True:
            maximum = np.max(img)
            minimum = np.min(img)
            img = (img - minimum) / (maximum - minimum)

        # calculate how many cubes are needed for each dimension
        x_range = math.ceil(img.shape[2] / self.cube_size)
        y_range = math.ceil(img.shape[1] / self.cube_size)
        z_range = math.ceil(img.shape[0] / self.cube_size)
        self.number_cubes = x_range * y_range * z_range

        self.small_size = (int(self.cube_size/4), int(self.cube_size/4))
        self.big_size = (self.cube_size, self.cube_size)

        self.result_image = np.zeros(shape=(img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
        self.image_x = np.zeros(shape=(self.number_cubes, self.big_size[0], self.small_size[0], self.small_size[0]),
                                dtype=np.float32)
        self.image_y = np.zeros(shape=(self.number_cubes, self.big_size[0], self.small_size[0], self.small_size[0]),
                                dtype=np.float32)
        self.image_z = np.zeros(shape=(self.number_cubes, self.big_size[0], self.small_size[0], self.small_size[0]),
                                dtype=np.float32)

        index = 0
        for z in range(0, z_range):
            # first case of each if-else is the repositioning for the boundary cubes
            if z * self.cube_size + self.cube_size > img.shape[0]:
                zi = img.shape[0] - self.cube_size
            else:
                zi = z * self.cube_size

            for y in range(0, y_range):
                if y * self.cube_size + self.cube_size > img.shape[1]:
                    yi = img.shape[1] - self.cube_size
                else:
                    yi = y * self.cube_size

                for x in range(0, x_range):
                    if x * self.cube_size + self.cube_size > img.shape[2]:
                        xi = img.shape[2] - self.cube_size
                    else:
                        xi = x * self.cube_size

                    # get the partial image and rotate it to cover dimension x,y and z
                    temp_x = img[zi:zi + self.cube_size, yi:yi + self.cube_size, xi:xi + self.cube_size]
                    temp_y = np.rot90(img[zi:zi + self.cube_size, yi:yi + self.cube_size, xi:xi + self.cube_size],
                                      1, axes=(0, 1))
                    temp_z = np.rot90(img[zi:zi + self.cube_size, yi:yi + self.cube_size, xi:xi + self.cube_size],
                                      1, axes=(0, 2))

                    # resize all images and set values to range -1, 1, and fill the cube
                    for i in range(0, self.cube_size):
                        self.image_x[index, i, :, :] = rescale(temp_x[i, :, :], 0.25, anti_aliasing=False)
                        self.image_x[index, i, :, :] = (self.image_x[index, i, :, :] * 2) - 1

                        self.image_y[index, i, :, :] = rescale(temp_y[i, :, :], 0.25, anti_aliasing=False)
                        self.image_y[index, i, :, :] = (self.image_y[index, i, :, :] * 2) - 1

                        self.image_z[index, i, :, :] = rescale(temp_z[i, :, :], 0.25, anti_aliasing=False)
                        self.image_z[index, i, :, :] = (self.image_z[index, i, :, :] * 2) - 1

                    # for result image to access the position for each cube
                    self.positions.append((zi, yi, xi))
                    index += 1

    def provide_test_3dcube(self, dim):
        """
            Prepares image and its mask that are then given
            to the neural network to be processed.
            ----------
            dim : string, dimension that is being processed
                        either x, y or z
            ----------
            x : array, contains the images
            m : array, contains the corresponding mask
        """
        if dim == "x":
            image = self.image_x
        elif dim == "y":
            image = self.image_y
        else:
            image = self.image_z
        x = np.zeros(shape=(self.big_size[0], self.small_size[0], self.small_size[0], self.num_channel),
                     dtype=np.float32)
        x[:, :, :, 0] = image[self.index]

        m = np.ones(shape=(self.big_size[0], self.big_size[0], self.big_size[1], self.num_class), dtype=np.bool)
        m[:, :, :, 1:] = m[:, :, :, 0:1]

        return x, m

    def init_test_image(self, image_path):
        """
            Reads in the image slice by slice and handles
            it like a 2D image.
            ----------
            image_patch : string, path of the image to be segmented
        """
        img = io.imread(image_path)

        if self.normalization is True:
            maximum = np.max(img)
            minimum = np.min(img)
            img = (img - minimum) / (maximum - minimum)

        # get x dimension and check of it is a multiple of 4, else add factor
        # TODO: assumed that x and y have the same size for factor calculation
        if img.shape[2] % 4 != 0:
            self.factor4 = 4 - img.shape[2] % 4
        self.big_size = (img.shape[2] + self.factor4, img.shape[1] + self.factor4)
        self.small_size = (int(self.big_size[0] / 4), int(self.big_size[1] / 4))
        self.stack_size = img.shape[0]

        self.result_image = np.zeros(shape=(img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
        self.image = np.zeros(shape=(img.shape[0], self.small_size[0], self.small_size[1]), dtype=np.float32)

        for i in range(0, img.shape[0]):
            # shrink the image by the factor 4
            self.image[i] = resize(img[i], (self.small_size[0], self.small_size[1]), anti_aliasing=False)
            self.image[i] = (self.image[i] * 2) - 1

    def provide_test(self):
        """
            Prepares image and its mask that are then given
            to the neural network to be processed.
            ------------
            x : array, contains the images
            m : array, contains the corresponding mask
        """
        x = np.zeros(shape=(1, self.small_size[0], self.small_size[1], self.num_channel), dtype=np.float32)
        x[:, :, :, 0] = self.image[self.index]

        # create an appropriate mask of size is not a multiple of 4
        m_temp = np.ones(shape=(1, self.big_size[0]-self.factor4, self.big_size[1]-self.factor4,
                                self.num_class), dtype=np.bool)
        m = np.zeros(shape=(1, self.big_size[0], self.big_size[1], self.num_class), dtype=np.bool)
        m[:, :self.big_size[0]-self.factor4, :self.big_size[1]-self.factor4, :] = m_temp
        m[:, :, :, 1:] = m[:, :, :, 0:1]
        self.index = self.index + 1

        return x, m
