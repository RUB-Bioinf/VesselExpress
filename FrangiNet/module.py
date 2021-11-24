import logging
import subprocess

import psutil
import tensorflow as tf
import numpy as np
from skimage import io
import os

from FrangiNet.franginet_model import FrangiNet
from FrangiNet.franginet_layer import pixel_wise_softmax_2
from FrangiNet.module_provider import DataProvider
from FrangiNet.graph_computation_thread import GraphComputationThread
from FrangiNet.guided_filter_layer import build_lr_can, deef_guided_filter_advanced

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

compThread = GraphComputationThread()


def main_FN_test(image, o, model, mode, normalization, average, mode_img, gpus, batch_size):
    # only used for calculating amount of memory
    start_mem = psutil.virtual_memory()[3]
    start_percent = psutil.virtual_memory()[2]
    """
        Main Function of FrangiNet.
        Covers the initialization of the neural net and
        the variables by calling the appropriate functions.
        ----------
        i : string, path of the image to be segmented
        o : string, path of the image containing the
            probability of each pixel to either be a vessel
            or background
        model : string, path to the trained model
        mode : string, "vanilla" for original version, "guided"
            using an additional layer to reduce noise
        normalization : boolean, normalize image before processing
        mode_img: string, "3DCube" for splitting the image in
            cubes to be processed in all three dimensions, "OneCube"
            to use a single cube to be processed in all three
            dimensions, "Default" handles the image as single
            2D images that are stacked after being processed
    """

    if normalization == "True":
        normalization = True
    else:
        normalization = False
    if average == "True":
        average = True
    else:
        average = False
    batch_size = int(batch_size)

    data = DataProvider(image, normalization, mode_img, batch_size)  # initialize all the needed variables

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    conf = tf.ConfigProto()  # configuration options for a session
    conf.gpu_options.per_process_gpu_memory_fraction = 0.99
    conf.gpu_options.allow_growth = True

    if mode == 'vanilla':
        preprocessed = data.x
    else:  # preprocess images with guided filter layer to reduce noise
        inputs = data.x
        guid = build_lr_can(inputs=inputs, is_training=data.is_training)
        guid = 2.0 * (tf.nn.sigmoid(guid) - 0.5)
        preprocessed = deef_guided_filter_advanced(inputs=inputs, guid=guid, is_training=data.is_training, r=1,
                                                   eps=1e-8)

    # logits: raw predictions as outcome from last layer of neural network
    # fed into softmax to get probabilities for the predicted classes
    fraggi = FrangiNet()
    fraggi_logits = fraggi.multi_scale(inputs=preprocessed, is_training=data.is_training)
    fraggi_softmax = pixel_wise_softmax_2(fraggi_logits, name='fraggi_softmax')
    ups = tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last', name='up')
    fraggi_softmax = ups(fraggi_softmax)
    fraggi_softmax_out = fraggi_softmax * tf.cast(data.m, dtype=tf.float32)

    with tf.Session(config=conf) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model)
        logging.info("Model restored from file: {:}".format(model))

        if data.mode == "3DCube":
            for index in range(0, data.number_cubes):
                x_test, m_test = data.provide_test_3dcube("x")
                feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
                pred_x = sess.run(fraggi_softmax_out, feed_dict=feed_dict)

                x_test, m_test = data.provide_test_3dcube("y")
                feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
                pred_y = sess.run(fraggi_softmax_out, feed_dict=feed_dict)

                x_test, m_test = data.provide_test_3dcube("z")
                feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
                pred_z = sess.run(fraggi_softmax_out, feed_dict=feed_dict)

                # rotate y and z dimension back
                rot_y = np.rot90(pred_y[:, :, :, -1], -1, axes=(0, 1))
                rot_z = np.rot90(pred_z[:, :, :, -1], -1, axes=(0, 2))

                # put cube back in the right position of the result image
                data.result_image[data.positions[index][0]:data.positions[index][0] + data.cube_size,
                data.positions[index][1]:data.positions[index][1] + data.cube_size,
                data.positions[index][2]:data.positions[index][2] + data.cube_size] \
                    = np.stack([pred_x[:, :, :, -1], rot_y[:, :, :], rot_z[:, :, :]], axis=0).max(axis=0)

                '''Calculate average of all three dimensions
                data.result_image[data.positions[index][0]:data.positions[index][0] + data.cube_size,
                data.positions[index][1]:data.positions[index][1] + data.cube_size,
                data.positions[index][2]:data.positions[index][2] + data.cube_size] \
                    = pred_x[:, :, :, -1] / 3 + rot_y[:, :, :] / 3 + rot_z[:, :, :] / 3'''

                data.index = data.index + 1
                print("Finished Cube " + str(index+1) + " of " + str(data.number_cubes))

        elif data.mode == "OneCube":
            x_test, m_test = data.provide_test_onecube("x")
            feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
            v_p = sess.run(fraggi_softmax_out, feed_dict=feed_dict)

            image_x = v_p[:data.dim_xyz[2], :data.dim_xyz[1], :data.dim_xyz[0], -1]
            print("Finished Cube processing x dimension")

            x_test, m_test = data.provide_test_onecube("y")
            feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
            v_p = sess.run(fraggi_softmax_out, feed_dict=feed_dict)

            image_y = np.zeros(shape=(data.dim_xyz[2], data.dim_xyz[1], data.dim_xyz[0]), dtype=np.float32)
            for i in range(0, data.dim_xyz[1]):  # "rotate" y dimension
                image_y[:data.dim_xyz[2], i, :data.dim_xyz[0]] = v_p[i, :data.dim_xyz[2], :data.dim_xyz[0], -1]
            print("Finished Cube processing y dimension")

            x_test, m_test = data.provide_test_onecube("z")
            feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
            v_p = sess.run(fraggi_softmax_out, feed_dict=feed_dict)

            image_z = np.zeros(shape=(data.dim_xyz[2], data.dim_xyz[1], data.dim_xyz[0]), dtype=np.float32)
            for i in range(0, data.dim_xyz[0]):
                image_z[:data.dim_xyz[2], :data.dim_xyz[1], i] = v_p[i, :data.dim_xyz[2], :data.dim_xyz[1], -1]
            print("Finished Cube processing z dimension")

            if average is True:
                print("Calculate average value")
                data.result_image = image_x / 3 + image_y / 3 + image_z / 3
            else:
                print("Calculate maximum value")
                data.result_image = np.stack([image_x, image_y, image_z], axis=0).max(axis=0)

        elif data.mode == "OneCubeBatch":
            image_x = np.zeros(shape=(data.dim_xyz[2], data.dim_xyz[1], data.dim_xyz[0]), dtype=np.float32)
            for ind in range(0, data.dim_xyz[2] // data.batch_size + (data.dim_xyz[2] % data.batch_size > 0)):
                x_test, m_test = data.provide_test_onecube_batch("x")
                feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
                v_p = sess.run(fraggi_softmax_out, feed_dict=feed_dict)
                image_x[data.index - data.cur_size:data.index, :data.dim_xyz[0], :data.dim_xyz[1]] = \
                    v_p[:data.cur_size, :data.dim_xyz[0], :data.dim_xyz[1], -1]

            print("Finished Cube processing x dimension")
            data.index = 0

            image_y = np.zeros(shape=(data.dim_xyz[2], data.dim_xyz[1], data.dim_xyz[0]), dtype=np.float32)
            for ind in range(0, data.dim_xyz[1] // data.batch_size + (data.dim_xyz[1] % data.batch_size > 0)):
                x_test, m_test = data.provide_test_onecube_batch("y")
                feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
                v_p = sess.run(fraggi_softmax_out, feed_dict=feed_dict)
                for i in range(0, data.cur_size):  # "rotate" y dimension
                    image_y[:data.dim_xyz[2], data.index - data.cur_size + i, :data.dim_xyz[0]] = \
                        v_p[i, :data.dim_xyz[2], :data.dim_xyz[0], -1]

            print("Finished Cube processing y dimension")
            data.index = 0

            image_z = np.zeros(shape=(data.dim_xyz[2], data.dim_xyz[1], data.dim_xyz[0]), dtype=np.float32)
            for ind in range(0, data.dim_xyz[0] // data.batch_size + (data.dim_xyz[0] % data.batch_size > 0)):
                x_test, m_test = data.provide_test_onecube_batch("z")
                feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
                v_p = sess.run(fraggi_softmax_out, feed_dict=feed_dict)
                for i in range(0, data.cur_size):
                    image_z[:data.dim_xyz[2], :data.dim_xyz[1], data.index - data.cur_size + i] = \
                        v_p[i, :data.dim_xyz[2], :data.dim_xyz[1], -1]
            print("Finished Cube processing z dimension")

            if average is True:
                print("Calculate average value")
                data.result_image = image_x / 3 + image_y / 3 + image_z / 3
            else:
                print("Calculate maximum value")
                data.result_image = np.stack([image_x, image_y, image_z], axis=0).max(axis=0)

        else:
            for index in range(0, data.stack_size):
                x_test, m_test = data.provide_test()
                feed_dict = {data.x: x_test, data.m: m_test, data.is_training: False}
                v_p = sess.run(fraggi_softmax_out, feed_dict=feed_dict)
                data.result_image[index] = v_p[0, :data.big_size[0] - data.factor4, :data.big_size[1] - data.factor4, -1]

        sess.close()

    print("memory used: " + str((psutil.virtual_memory()[3] - start_mem) / 10**9) + " GB (" + str(psutil.virtual_memory()[2] - start_percent) + " %)")
    f = open(os.path.dirname(os.path.abspath(image).replace('\\', '/')) + '/measurement.csv', 'a')
    f.write("Memory used (Frangi-Filter);" + "{:.3f}".format((psutil.virtual_memory()[3] - start_mem) / 10**9) + ";GB\n")
    f.close()

    io.imsave(o, data.result_image)  # give back save path and image with probabilities
