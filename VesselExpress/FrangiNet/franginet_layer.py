# Weilin Fu (2019) Frangi-Net on High-Resolution Fundus (HRF) image database [Source Code].
# https://doi.org/10.24433/CO.5016803.v2
# GNU General Public License (GPL) */

import tensorflow as tf


def conv2d(x, w, padding='VALID', keep_prob=1, name='conv'):
    with tf.name_scope(name): # context manager pushes name scope, makes name of operations added within have a prefix.
        conv = tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding=padding)
        return tf.nn.dropout(conv, keep_prob)
        # tf.nn: wrappers for primitive Neural Net (NN) Operations
        # conv2d: computes a 2-D convolution given input and 4-D filters tensors
        # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
        # dropout: randomly sets elements to zero to prevent overfitting
        # https://www.tensorflow.org/api_docs/python/tf/nn/dropout


def elementwise2d(l1, l2, op, coefficient=[1., 1.], name='elementwise'):
    with tf.name_scope(name):
        t1 = tf.multiply(l1, coefficient[0])
        t2 = tf.multiply(l2, coefficient[1])
        # different operations for two tensors, also maximum of the two as well as getting the greater absolute value
        if op == 'add':
            return tf.add(t1, t2)
        elif op == 'subtract':
            return tf.subtract(t1, t2)
        elif op == 'divide':
            return tf.div(t1, t2)
        elif op == 'multiply':
            return tf.multiply(t1, t2)
        elif op == 'max':
            return tf.maximum(t1, t2)
        elif op == 'max_abs':
            return tf.where(condition=tf.abs(t1) > tf.abs(t2), x=t1, y=t2)
        elif op == 'min_abs':
            return tf.where(condition=tf.abs(t1) < tf.abs(t2), x=t1, y=t2)
        else:
            return None


def nonlinear2d(var, op, name='nonlinear'):
    with tf.name_scope(name):
        if op == 'square':
            return elementwise2d(var, var, 'multiply')
        elif op == 'square_root':
            return tf.sqrt(var)
        elif op == 'exp':
            return tf.exp(var)
        else:
            return None


def elementwise_sum(l1, l2, l3, coefficient=[1., 1., 1.], name='elementwise_sum'):
    with tf.name_scope(name):
        t1 = tf.multiply(l1, coefficient[0])
        t2 = tf.multiply(l2, coefficient[1])
        t3 = tf.multiply(l3, coefficient[2])
        return tf.add(tf.add(t1, t2), t3)


def condition_operator(x, y, op='less', name=''):
    with tf.name_scope(name):
        if op == 'less':
            return tf.less(y=y, x=x, name='less_mask')


def batch_norm_tensor(inputs, is_training, scope, moments_dims=[0, 1, 2], bn_decay=.95):
    with tf.variable_scope(scope):
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')  # calculates mean/variance of inputs
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        # operator that maintains moving averages of variables
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # update moving average and return current batch's average and variance
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average return the variable holding the average of variance
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def pixel_wise_softmax_2(output_map, name):
    with tf.variable_scope(name):
        exponential_map = tf.exp(output_map)
        sum_exp = tf.reduce_sum(exponential_map, 3, keepdims=True)
        tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
        return tf.div(exponential_map, tensor_sum_exp)
