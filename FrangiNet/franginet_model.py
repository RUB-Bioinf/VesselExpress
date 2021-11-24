# Weilin Fu (2019) Frangi-Net on High-Resolution Fundus (HRF) image database [Source Code].
# https://doi.org/10.24433/CO.5016803.v2
# GNU General Public License (GPL)

import tensorflow as tf
from FrangiNet.franginet_layer import conv2d, elementwise2d, nonlinear2d, elementwise_sum, condition_operator, batch_norm_tensor
import numpy as np
import scipy.ndimage as ndi
from itertools import combinations_with_replacement
from tensorflow.python.layers.convolutional import conv3d


class FrangiNet:
    def __init__(self):
        self.sigma_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.init_kernels()
        self.raw_vesselness = None

    def init_kernels(self):
        self.w_xx_list = []
        self.w_yy_list = []
        self.w_xy_list = []
        for i in range(len(self.sigma_list)):
            sigma = self.sigma_list[i]

            # calculate gaussian for the current kernel with the current sigma
            size = int(sigma * 6 + 1)
            if size % 2 == 0:
                size += 1
            x = np.zeros((size, size))
            x[int((size - 1) / 2), int((size - 1) / 2)] = 1

            # calculate second deviation
            img = np.zeros(shape=(size, size), dtype=np.float32)
            miu = int((size - 1) / 2)
            img[miu, miu] = 1
            axes = range(img.ndim)
            gf = ndi.gaussian_filter(img, sigma, mode='constant')
            gradients = np.gradient(gf)
            xx2d, xy2d, yy2d = [np.gradient(gradients[ax0], axis=ax1)
                          for ax0, ax1 in combinations_with_replacement(axes, 2)]

            # build the hessian matrix needed for the frangi filter
            xx_mat = np.zeros((size, size, 1, 1), dtype=np.float32)
            yy_mat = np.zeros((size, size, 1, 1), dtype=np.float32)
            xy_mat = np.zeros((size, size, 1, 1), dtype=np.float32)
            xx_mat[:, :, 0, 0] = xx2d
            yy_mat[:, :, 0, 0] = yy2d
            xy_mat[:, :, 0, 0] = xy2d
            w_xx = tf.get_variable(initializer=xx_mat, dtype=tf.float32, name='w_xx_%d' % i, trainable=True)
            w_yy = tf.get_variable(initializer=yy_mat, dtype=tf.float32, name='w_yy_%d' % i, trainable=True)
            w_xy = tf.get_variable(initializer=xy_mat, dtype=tf.float32, name='w_xy_%d' % i, trainable=True)

            self.w_xx_list.append(w_xx)
            self.w_yy_list.append(w_yy)
            self.w_xy_list.append(w_xy)

    def single_scale(self, patch, scale_index, name=''):
        sigma = self.sigma_list[scale_index]
        w_xx = self.w_xx_list[scale_index]
        w_yy = self.w_yy_list[scale_index]
        w_xy = self.w_xy_list[scale_index]

        # computes a 2-D convolution
        conv_xx = conv2d(patch, w_xx, 'SAME', name=name+'conv_xx')
        conv_yy = conv2d(patch, w_yy, 'SAME', name=name+'conv_yy')
        conv_xy = conv2d(patch, w_xy, 'SAME', name=name+'conv_xy')

        # Returns an element-wise 2D convolution * sigma²
        scale_xx = tf.multiply(conv_xx, sigma**2, name=name+'scale_xx')
        scale_yy = tf.multiply(conv_yy, sigma**2, name=name+'scale_yy')
        scale_xy = tf.multiply(conv_xy, sigma**2, name=name+'scale_xy')

        # different calculations for variables for frangi filter
        xx_yy = elementwise2d(scale_xx, scale_yy, 'subtract', name=name+'xx_yy')
        xx_yy_sqr = nonlinear2d(xx_yy, 'square', name=name+'xx_yy_sqr')
        xy_sqr = nonlinear2d(scale_xy, 'square', name=name+'xy_sqr')
        b2_4ac = elementwise2d(xx_yy_sqr, xy_sqr, 'add', coefficient=[1., 4.], name=name+'b2_4ac')
        b2_4ac_sqrt = nonlinear2d(b2_4ac, 'square_root', name=name+'b2_4ac_sqrt')
        root1 = elementwise_sum(scale_xx, scale_yy, b2_4ac_sqrt, coefficient=[0.5, 0.5, 0.5], name=name+'root1')
        root2 = elementwise_sum(scale_xx, scale_yy, b2_4ac_sqrt, coefficient=[0.5, 0.5, -0.5], name=name+'root2')

        # variables for frangi and calculation (see paper and the equation on side 2)
        lambda1 = elementwise2d(root1, root2, 'max_abs', name=name+'lambda1')
        lambda2 = elementwise2d(root1, root2, 'min_abs', name=name+'lambda2')
        mask = condition_operator(x=[0.], y=lambda1, op='less', name=name+'less_mask')

        lambda1_sqr = nonlinear2d(lambda1, 'square', name=name+'lambda1_sqr')
        lambda2_sqr = nonlinear2d(lambda2, 'square', name=name+'lambda2_sqr')
        rb_sqr = elementwise2d(lambda2_sqr, lambda1_sqr, 'divide', name=name+'rb_sqr')
        s_sqr = elementwise2d(lambda1_sqr, lambda2_sqr, 'add', name=name+'s_sqr')
        a = tf.Variable(dtype=tf.float32, initial_value=-2.)
        b = tf.Variable(dtype=tf.float32, initial_value=-0.5)

        rb_sqr_scale = elementwise2d(rb_sqr, a, op='multiply')
        s_sqr_scale = elementwise2d(s_sqr, b, op='multiply')
        exp_rb_sqr = nonlinear2d(rb_sqr_scale, op='exp')
        exp_s_sqr = nonlinear2d(s_sqr_scale, op='exp')
        v_before_mask = elementwise2d(exp_rb_sqr, elementwise2d(exp_rb_sqr, exp_s_sqr, op='multiply'), op='subtract')
        zeros = tf.zeros(tf.shape(v_before_mask))

        vesselness = tf.where(condition=mask, x=v_before_mask, y=zeros)
        with tf.variable_scope('parameters'):
            tf.summary.scalar(name+'/a', a)
            tf.summary.scalar(name+'/b', b)

        return vesselness, conv_xx

    def multi_scale(self, inputs, is_training):
        for i in range(len(self.sigma_list)):
            if i == 0:
                vesselness, conv_xx = self.single_scale(inputs, i, name='scale%d' % i)
            else:
                temp, conv_xx = self.single_scale(inputs, 1, name='scale%d' % i)
                vesselness = tf.maximum(vesselness, temp)

        self.raw_vesselness = vesselness

        w1 = tf.get_variable(shape=(1, 1, 1, 2), dtype=tf.float32, name='final/w1')
        w2 = tf.get_variable(shape=(1, 1, 2, 2), dtype=tf.float32, name='final/w2')
        b1 = tf.Variable(initial_value=tf.constant(0.1, shape=[2]), name='final/b1')
        b2 = tf.Variable(initial_value=tf.constant(0.1, shape=[2]), name='final/b2')

        vesselness = conv2d(vesselness, w1, padding='SAME', name='final/conv1') + b1
        vesselness = batch_norm_tensor(vesselness, is_training=is_training, scope='final/bn1')
        vesselness = conv2d(vesselness, w2, padding='SAME', name='final/conv2') + b2

        return vesselness
