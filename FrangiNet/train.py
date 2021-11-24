import logging
import os

import tensorflow as tf
from FrangiNet import metrics
from FrangiNet.metrics import Metrics
from FrangiNet.train_provider import DataProvider
from FrangiNet.franginet_model import FrangiNet
from FrangiNet.franginet_layer import pixel_wise_softmax_2
from FrangiNet.franginet_loss import get_loss
from FrangiNet.guided_filter_layer import build_lr_can, deef_guided_filter_advanced
from FrangiNet.graph_computation_thread import GraphComputationThread
from skimage import io
import numpy as np
import time
from tensorflow.python.client import device_lib

compThread = GraphComputationThread()  # used for the session/training
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def main_frangi_train(conf_file):
    """
        Main Function of FrangiNet.
        Covers the initialization of the neural net and
        the variables by calling the appropriate functions.
        ----------
        conf_file : string, path of the json file which contains
            all necessary parameters to be set by the user.
    """
    # add the values from the config file into the model and load the image into
    data = DataProvider(conf_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = data.gpus
    # print(device_lib.list_local_devices())
    # print(tf.__version__)

    # initialize the values and path for the metrics
    metric = Metrics(data.path_save, data.binary_image, data.valid_binary_image, data.is_restore)

    if data.mode != 'vanilla':
        global compThread
        conf = tf.ConfigProto()  # configuration options for a session
        conf.gpu_options.per_process_gpu_memory_fraction = 0.5

    if data.mode == 'vanilla':
        preprocessed = data.x
    else:  # preprocess images with guided filter layer to reduce noise
        inputs = data.x
        guid = build_lr_can(inputs=inputs, is_training=data.is_training)
        guid = 2.0 * (tf.nn.sigmoid(guid) - 0.5)
        preprocessed = deef_guided_filter_advanced(inputs=inputs, guid=guid, is_training=data.is_training,
                                                   r=1, eps=1e-8)

    def up_sample(inp):  # resize smaller image up again
        ups = tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last', name='up')
        outputs = ups(inp)
        return outputs

    # logits: raw predictions as outcome from last layer of neural network
    # fed into softmax to get probabilities for the predicted classes
    fraggi = FrangiNet()
    logits = fraggi.multi_scale(inputs=preprocessed, is_training=data.is_training)
    softmax = pixel_wise_softmax_2(logits, name='softmax')
    logits = up_sample(logits)  # sample x up to a 500, 500 tensor from 125, 125
    softmax = up_sample(softmax)

    if data.mode == 'vanilla':
        loss_wcr, loss_fc = get_loss(logits, data.y, data.m, data.w)
        total_loss = loss_fc
    else:
        loss_sim = tf.losses.mean_squared_error(predictions=tf.boolean_mask(up_sample(preprocessed),
                                                                            data.m[:, :, :, -1:]),
                                                labels=tf.boolean_mask(up_sample(data.x),
                                                                       data.m[:, :, :, -1:]))
        loss_wcr, loss_fc = get_loss(logits, data.y, data.m, data.w)
        total_loss = loss_fc + data.reg_sim * loss_sim
    # loss_wcr: tf.nn.weighted_cross_entropy_with_logits, added with weight and reduce_mean

    # lower training rate as training progresses
    learning_rate_node = tf.train.exponential_decay(learning_rate=data.rate_learning,
                                                    global_step=data.step_global,
                                                    decay_steps=data.step_decay,
                                                    decay_rate=data.rate_decay,
                                                    staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_node).minimize(total_loss, name='ADAM',
                                                                                   global_step=data.step_global)

    # save model
    saver = tf.train.Saver(max_to_keep=10000000)
    sess = tf.Session()

    if data.is_restore:
        saver.restore(sess, data.path_model % data.step_restore)
        logging.info("Model restored from file: {:}".format(data.path_model % data.step_restore))
        total_step = data.steps_per_epoch * data.step_restore
    else:
        sess.run(tf.global_variables_initializer())
        total_step = 0

    compThread.setSession(sess)  # set the tensorflow model (sess) as a thread

    if data.early_stopping is True:
        epoch_loss = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 100000000000]

    while total_step <= data.step_overall:
        if total_step % data.steps_per_epoch == 0:
            time_train_start = time.time()
        total_step = total_step + 1

        # get the training data (image, label, mask, weight)
        x_data, y_data, m_data, w_data = data.provide_train()
        feed_dict = {data.x: x_data, data.y: y_data, data.m: m_data, data.w: w_data,
                     data.is_training: True}

        if data.mode == 'vanilla':
            pred_train, loss_train, _ = sess.run([softmax, loss_fc, train_step], feed_dict=feed_dict)
        else:
            compThread.setParameters([softmax, total_loss, train_step], feed_dict)
            compThread.start()
            compThread.join()

            # compute the new values and loss for the current batch
            pred_train, loss_train, _ = compThread.getResult()

        if total_step % data.steps_per_epoch == 0:
            logging.info('Training epoch %s finished, loss: %s', str(int(total_step / data.steps_per_epoch)),
                         str(loss_train))
            time_train_end = time.time()
            '''f = open('FrangiNet/Time_Training.csv', 'a')
            f.write(str(int(total_step / data.steps_per_epoch)) + ";" + "{:.3f}".format(
                time_train_end - time_train_start) + ";s\n")
            f.close()'''

        # validation is run for every epoch
        if total_step % data.steps_per_epoch == 0:

            # prepare validation images for calculating metrics
            if total_step % data.step_summary == 0:
                pred_valid = np.zeros(shape=(data.valid_number_slices, data.image_big[0],
                                             data.image_big[1]), dtype=np.float32)
                label_valid = np.zeros(shape=(data.valid_number_slices, data.image_big[0],
                                              data.image_big[1]), dtype=np.float32)
                index = 0

            time_valid_start = time.time()
            for _ in range(0, data.steps_per_validation):
                x_bv, y_bv, m_bv, w_bv = data.provide_valid()
                feed_dict_bv = {data.x: x_bv, data.y: y_bv, data.m: m_bv, data.w: w_bv,
                                data.is_training: False}
                pred, crop_x, crop_y, crop_m, crop_w, loss_valid = sess.run([softmax, data.x, data.y,
                                                                             data.m, data.w, total_loss], feed_dict_bv)

                # fill in the prepared validation images for calculating metrics
                if total_step % data.step_summary == 0:
                    pred_valid[index * data.size_batch:index * data.size_batch +
                                                       data.size_batch, :, :] = pred[:, :, :, -1]
                    label_valid[index * data.size_batch:index * data.size_batch +
                                                        data.size_batch, :, :] = crop_y[:, :, :, -1]
                    index += 1
            logging.info('Validation epoch %s finished, loss: %s', str(int(total_step / data.steps_per_epoch)),
                         str(loss_valid))
            time_valid_end = time.time()
            '''f = open('FrangiNet/Time_Validation.csv', 'a')
            f.write(str(int(total_step / data.steps_per_epoch)) + ";" + "{:.3f}".format(
                time_valid_end - time_valid_start) + ";s\n")
            f.close()'''

        # collect the metrics of the training data set for the whole epoch by taking every batch of said epoch
        if total_step % data.step_summary > data.step_summary - data.steps_per_epoch:
            metric.collect_train_epoch(pred_train[:, :, :, -1], data.current_images)

        if total_step % data.step_summary == 0:
            metric.collect_train_epoch(pred_train, data.current_images)
            metric.metrics_to_csv(int(total_step / data.steps_per_epoch), loss_train, loss_valid, pred_valid,
                                  label_valid)

        # save the model and an example of an image including its label and prediction
        # also create roc and precision-recall-curve of the validation set
        if total_step % data.step_save == 0:
            time_save_start = time.time()
            model_path = data.path_save + 'model/%d/' % int(total_step / data.steps_per_epoch)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            saver.save(sess, model_path + 'model.ckpt')

            io.imsave(model_path + '/pred.tif', pred[0, :, :, -1], check_contrast=False)
            io.imsave(model_path + '/img.tif', crop_x[0, :, :, 0], check_contrast=False)
            io.imsave(model_path + '/label.tif', crop_y[0, :, :, -1], check_contrast=False)

            if data.restrict_roc is True:
                if data.create_prc is True:
                    metrics.precision_recall(pred_valid[:data.restrict_amount, :, :],
                                             label_valid[:data.restrict_amount, :, :],
                                             int(total_step / data.steps_per_epoch), data.path_save, model_path,
                                             data.threshold_metric)
                if data.create_roc is True:
                    metrics.roc(pred_valid[:data.restrict_amount, :, :], label_valid[:data.restrict_amount, :, :],
                                int(total_step / data.steps_per_epoch), data.path_save)
            else:
                if data.create_prc is True:
                    metrics.precision_recall(pred_valid, label_valid, int(total_step / data.steps_per_epoch),
                                             data.path_save, model_path, data.threshold_metric)
                if data.create_roc is True:
                    metrics.roc(pred_valid, label_valid, int(total_step / data.steps_per_epoch), data.path_save)
            metrics.create_metrics_from_csv(data.path_save)

            logging.info('Model epoch %s saved and metrics created', str(int(total_step / data.steps_per_epoch)))
            time_save_end = time.time()
            '''f = open('FrangiNet/Time_Saving.csv', 'a')
            f.write(str(int(total_step / data.steps_per_epoch)) + ";" + "{:.3f}".format(time_save_end - time_save_start)
                    + ";s\n")
            f.close()'''

        # ### EARLY STOPPING
        if data.early_stopping is True and total_step % data.steps_per_epoch == 0:
            # get loss from the last 10 epochs and see if less than (chosen) delta
            epoch_loss.append(loss_train)
            epoch_loss.pop(0)
            if sum(epoch_loss) / 10 < data.delta_loss:
                if total_step % data.step_save == 0:
                    # save the current model and stop training
                    model_path = data.path_save + 'model/%d/' % int(total_step / data.steps_per_epoch)
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    saver.save(sess, model_path + 'model.ckpt')

                    parameters = [softmax, data.x, data.y, data.m, data.w]
                    pred_log, crop_x, crop_y, crop_m, crop_w = sess.run(parameters, feed_dict_bv)

                    if data.create_prc is True:
                        metrics.precision_recall(pred_log, crop_y, int(total_step / data.steps_per_epoch),
                                                 data.path_save, model_path, data.threshold_metric)
                    if data.create_roc is True:
                        metrics.roc(pred_log, crop_y, int(total_step / data.steps_per_epoch), data.path_save)
                    metrics.create_metrics_from_csv(data.path_save)

                    logging.info('Model of epoch %s has been saved' % str(int(total_step / data.steps_per_epoch)))

                logging.info('Training has stopped early in epoch %s.' % str(int(total_step / data.steps_per_epoch)))
                break

        if np.isnan(loss_train):
            break

    sess.close()
