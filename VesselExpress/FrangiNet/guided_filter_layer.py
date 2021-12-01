# Weilin Fu (2019) Frangi-Net on High-Resolution Fundus (HRF) image database [Source Code].
# https://doi.org/10.24433/CO.5016803.v2
# GNU General Public License (GPL)

from guided_filter_tf.guided_filter import guided_filter
import tensorflow as tf
from FrangiNet.franginet_layer import batch_norm_tensor


def build_lr_can(inputs, is_training, kernel_size=3, num_channel=1, num_feature=8, num_layer=5, strides=[1, 1, 1, 1]):
    w1 = tf.get_variable(shape=(kernel_size, kernel_size, num_channel, num_feature), dtype=tf.float32, name='lrn/w1')
    inputs = tf.nn.conv2d(input=inputs, filter=w1, strides=strides, padding='SAME', name='lrn/conv1')
    inputs = batch_norm_tensor(inputs, is_training=is_training, scope='lrn/bn1')
    inputs = tf.nn.leaky_relu(inputs, alpha=0.2, name='lrn/l_relu1')
    for i in range(2, 1 + num_layer):
        w_i = tf.get_variable(shape=(kernel_size, kernel_size, num_feature, num_feature),
                              dtype=tf.float32, name='lrn/w%d' % i)
        inputs = tf.nn.atrous_conv2d(value=inputs, filters=w_i, rate=2**(i-1), padding='SAME', name='lrn/dil_conv%d' % i)
        inputs = batch_norm_tensor(inputs, is_training=is_training, scope='lrn/bn%d' % i)
        inputs = tf.nn.leaky_relu(inputs, alpha=0.2, name='lrn/l_relu%d' % i)

    w_final0 = tf.get_variable(shape=(kernel_size, kernel_size, num_feature, num_feature),
                               dtype=tf.float32, name='lrn/w_final0')
    inputs = tf.nn.conv2d(input=inputs, filter=w_final0, strides=strides, padding='SAME', name='lrn/conv_final0')
    inputs = batch_norm_tensor(inputs, is_training=is_training, scope='lrn/bn_final')
    inputs = tf.nn.leaky_relu(inputs, alpha=0.2, name='lrn/l_relu_final')
    w_final1 = tf.get_variable(shape=(1, 1, num_feature, num_channel), dtype=tf.float32, name='lrn/w_final1')
    outputs = tf.nn.conv2d(input=inputs, filter=w_final1, strides=strides, padding='SAME', name='lrn/conv_final1')

    return outputs


def deef_guided_filter_advanced(inputs, guid, is_training, r=1, eps=1e-8):
    def guided_map(inputs, kernel_size=1, num_channel=1, num_feature=5, strides=[1, 1, 1, 1], name='gm'):
        w1 = tf.get_variable(shape=(kernel_size, kernel_size, num_channel, num_feature), dtype=tf.float32,
                             name=name + '/w1')
        w2 = tf.get_variable(shape=(kernel_size, kernel_size, num_feature, num_channel), dtype=tf.float32,
                             name=name + '/w2')
        b2 = tf.Variable(initial_value=tf.constant(0.1, shape=[1]), name=name + '/bn2')
        inputs = tf.nn.conv2d(input=inputs, filter=w1, strides=strides, padding='SAME', name=name + '/conv1')
        inputs = batch_norm_tensor(inputs, is_training=is_training, scope=name + '/bn')
        inputs = tf.nn.leaky_relu(inputs, alpha=0.2)
        outputs = tf.nn.conv2d(input=inputs, filter=w2, strides=strides, padding='SAME', name=name + '/conv2') + b2

        return outputs

    inputs = guided_map(inputs=inputs, name='inputs_guided')
    outputs = guided_filter(x=inputs, y=guid, r=r, eps=eps, nhwc=True)
    return outputs
