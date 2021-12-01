import logging

import tensorflow as tf
import os
import time

from FrangiNet.franginet_model import FrangiNet
from FrangiNet.franginet_layer import pixel_wise_softmax_2
from FrangiNet.test_provider import DataProvider
from FrangiNet.graph_computation_thread import GraphComputationThread
from FrangiNet.guided_filter_layer import build_lr_can, deef_guided_filter_advanced
from FrangiNet.metrics import Metrics, precision_recall, roc

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

compThread = GraphComputationThread()


def main_frangi_test(config_file):
    # only used for calculating amount of memory
    """
        Main Function of FrangiNet.
        Covers the initialization of the neural net and
        the variables by calling the appropriate functions.
        ----------
        conf_file : string, path of the json file which contains
            all necessary parameters to be set by the user.
    """

    data = DataProvider(config_file)  # initialize all the needed variables
    metric = Metrics(data.path_save, None, None, False, True)

    os.environ["CUDA_VISIBLE_DEVICES"] = data.gpus

    conf = tf.ConfigProto()  # configuration options for a session
    conf.gpu_options.per_process_gpu_memory_fraction = 0.99
    conf.gpu_options.allow_growth = True

    if data.mode == 'vanilla':
        preprocessed = data.x
    else:  # preprocess images with guided filter layer to reduce noise
        inputs = data.x
        guid = build_lr_can(inputs=inputs, is_training=data.is_training)
        guid = 2.0 * (tf.nn.sigmoid(guid) - 0.5)
        preprocessed = deef_guided_filter_advanced(inputs=inputs, guid=guid, is_training=data.is_training, r=1,
                                                   eps=1e-8)

    def up_sample(inp):  # resize smaller image up again
        ups = tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last', name='up')
        outputs = ups(inp)
        return outputs

    # logits: raw predictions as outcome from last layer of neural network
    # fed into softmax to get probabilities for the predicted classes
    fraggi = FrangiNet()
    logits = fraggi.multi_scale(inputs=preprocessed, is_training=data.is_training)
    softmax = pixel_wise_softmax_2(logits, name='fraggi_softmax')
    ups = tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last', name='up')
    softmax = ups(softmax)
    softmax_out = softmax * tf.cast(data.m, dtype=tf.float32)

    with tf.Session(config=conf) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, data.path_model)
        logging.info("Model restored from file: {:}".format(data.path_model))

        print("Testing Data")
        start = time.time()
        for index in range(0, data.image.shape[0]):
            x_test, m_test = data.provide_test()
            feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
            v_p = sess.run(softmax_out, feed_dict=feed_dict)
            data.result_image[index] = v_p[0, :, :, -1]

        print("Testing Data completed in %0.3f seconds" % (time.time() - start))

        if data.create_prc is True:
            start = time.time()
            print("Creating Precision-Recall-Curve")
            precision_recall(data.result_image, data.binary_image, 0, data.path_save, data.path_save,
                                    data.threshold_metric)
            print("PRC completed in %0.3f seconds" % (time.time() - start))
        if data.create_roc is True:
            start = time.time()
            print("Creating ROC-Curve")
            roc(data.result_image, data.binary_image, 0, data.path_save)
            print("ROC completed in %0.3f seconds" % (time.time() - start))

        print("Creating Metrics")
        start = time.time()
        metric.metrics_test(data.result_image, data.binary_image)
        print("Metrics completed in %0.3f seconds" % (time.time() - start))

        sess.close()
