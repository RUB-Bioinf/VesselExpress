import os
import matplotlib.pyplot as plt
from sklearn.metrics import *
import numpy as np


class Metrics:
    def __init__(self, path, label_train, label_valid, is_restore, is_test=False):
        """
            Initializes all necessary variables for calculating and
            storing metrics of the training.
            ----------
            patch : string, path where the metric files shall be saved
            label_train : array, contains all labels of the training set
            label_valid : array, contains all labels of the validation set
            is_restore : boolean, declares if the model is restored from
                an earlier training session
        """
        self.save_path = path + "metrics/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not is_restore:
            if not is_test:
                f = open(self.save_path + 'metrics.csv', 'a')
                f.write("Step;Loss Train;Loss Valid;TP;TN;FP;FN;Accuracy;Precision;Sensitivity;Specificity;F1 Score;TP Val;"
                        "TN Val;FP Val;FN Val;Accuracy Valid;Precision Valid;Sensitivity Valid;Specificity Valid;"
                        "F1 Score Valid\n0;1;1;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0\n")
                f.close()
            else:
                f = open(self.save_path + 'metrics.csv', 'a')
                f.write(
                    "TP;TN;FP;FN;Accuracy;Precision;Sensitivity;Specificity;F1 Score;\n")
                f.close()
        self.epoch_train = [0, 0, 0, 0]  # collects all tp, tn, fp and fn of the whole epoch (training set)
        self.epoch_valid = [0, 0, 0, 0]  # see above but for the validation set
        self.label_train = label_train
        self.label_valid = label_valid
        self.threshold = 0.49  # original value of FrangiNet, considering there are only two classes

    def collect_train_epoch_3d(self, pred_train, y):
        tp, tn, fp, fn = self.calculate_confusion_matrix(pred_train, y)

        # store tp, tn, fp and fn of the current batch
        self.epoch_train[0] += tp
        self.epoch_train[1] += tn
        self.epoch_train[2] += fp
        self.epoch_train[3] += fn

    def collect_valid_epoch_3d(self, pred_valid, y):
        tp, tn, fp, fn = self.calculate_confusion_matrix(pred_valid, y)

        # store tp, tn, fp and fn of the current batch
        self.epoch_valid[0] += tp
        self.epoch_valid[1] += tn
        self.epoch_valid[2] += fp
        self.epoch_valid[3] += fn

    def collect_train_epoch(self, pred_train, batch_images):
        """
            Calculates and stores the confusion matrix of the current
            batch of the training set.
            ----------
            pred_train : array, contains the predicted values of all
                images in the current training batch
            batch_images : list, contains all numbers of the images in
                the current training batch
        """
        # get the binary masks of the images that are in the current batch
        prep_label = np.zeros(shape=(len(batch_images), pred_train.shape[1], pred_train.shape[2]),
                              dtype=np.bool)
        index = 0
        for i in batch_images:
            prep_label[index] = self.label_train[i]
            index += 1
        tp, tn, fp, fn = self.calculate_confusion_matrix(pred_train, prep_label)

        # store tp, tn, fp and fn of the current batch
        self.epoch_train[0] += tp
        self.epoch_train[1] += tn
        self.epoch_train[2] += fp
        self.epoch_train[3] += fn

    def calculate_confusion_matrix(self, prediction, label):
        """
            Computes the confusion matrix on basis of the prediction
            compared to the label
            ----------
            precition : array, contains the prediction of the neural
                network of given images
            label : array, contains the labels of said images
            ----------
            tp : int, amount of true positives
            tn : int, amount of true negatives
            fp : int, amount of false positives
            fn : int, amount of false negatives
        """
        pred_ones = self.threshold <= prediction
        label_ones = np.array(label, dtype=bool)

        pred_zeros = ~pred_ones
        label_zeros = ~label_ones

        part1 = pred_ones[label_ones == 1]
        part2 = part1[part1 == 1]
        tp = part2.sum()

        part1 = pred_zeros[label_zeros == 1]
        part2 = part1[part1 == 1]
        tn = part2.sum()

        part1 = pred_zeros[label_ones == 1]
        part2 = part1[part1 == 1]
        fn = part2.sum()

        part1 = pred_ones[label_zeros == 1]
        part2 = part1[part1 == 1]
        fp = part2.sum()

        return tp, tn, fp, fn

    def calculate_metrics(self, tp, tn, fp, fn):
        """
            Computes the accuracy, precision, sensitivity, specitifity
            and the F1-Score of a given confusion matrix
            ----------
            precition : array, contains the prediction of the neural
                network of given images
            label : array, contains the labels of said images
            ----------
            accuracy : float
            precision : float
            sensitivity : float
            specificity : float
            f1_score : float
        """
        if tp is 0 and tn is 0 and fp is 0 and fn is 0:
            return 0, 0, 0, 0, 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1_score = 2 * ((precision * sensitivity) / (precision + sensitivity))

        return accuracy, precision, sensitivity, specificity, f1_score

    def metrics_to_csv_3d(self, step, loss_total, loss_valid):
        """
            Collects the confusion matrix of the whole training epoch,
            as well as the validation set. Calculates the metrics and
            writes them into a csv-file (path configured in the json-file)
            ----------
            step : int, current epoch
            loss_total : float, loss of the most recent batch of
                the training set
            loss_valid : float, loss of the most recent batch of
                the validation set
            pred_valid : array, contains the prediction array of
                the whole validation epoch
            label_valid : array, contains the corresponding labels
                of the validation images
        """
        tp = self.epoch_train[0]
        tn = self.epoch_train[1]
        fp = self.epoch_train[2]
        fn = self.epoch_train[3]
        self.epoch_train = [0, 0, 0, 0]
        acc, prec, sens, spec, f1_score = self.calculate_metrics(tp, tn, fp, fn)

        tp_v = self.epoch_valid[0]
        tn_v = self.epoch_valid[1]
        fp_v = self.epoch_valid[2]
        fn_v = self.epoch_valid[3]
        self.epoch_valid = [0, 0, 0, 0]
        acc_v, prec_v, sens_v, spec_v, f1_score_v = self.calculate_metrics(tp_v, tn_v, fp_v, fn_v)

        f = open(self.save_path + 'metrics.csv', 'a')
        f.write(str(step) + ";" + str(loss_total) + ";" + str(loss_valid) + ";" + str(tp) + ";" + str(tn) + ";" + str(
            fp) + ";" +
                str(fn) + ";" + str(acc) + ";" + str(prec) + ";" + str(sens) + ";" + str(spec) + ";" + str(f1_score) +
                ";" + str(tp_v) + ";" + str(tn_v) + ";" + str(fp_v) + ";" + str(fn_v) + ";" + str(acc_v) + ";" +
                str(prec_v) + ";" + str(sens_v) + ";" + str(spec_v) + ";" + str(f1_score_v) + "\n")
        f.close()

    def metrics_to_csv(self, step, loss_total, loss_valid, pred_valid, label_valid):
        """
            Collects the confusion matrix of the whole training epoch,
            as well as the validation set. Calculates the metrics and
            writes them into a csv-file (path configured in the json-file)
            ----------
            step : int, current epoch
            loss_total : float, loss of the most recent batch of
                the training set
            loss_valid : float, loss of the most recent batch of
                the validation set
            pred_valid : array, contains the prediction array of
                the whole validation epoch
            label_valid : array, contains the corresponding labels
                of the validation images
        """
        tp = self.epoch_train[0]
        tn = self.epoch_train[1]
        fp = self.epoch_train[2]
        fn = self.epoch_train[3]
        self.epoch_train = [0, 0, 0, 0]
        acc, prec, sens, spec, f1_score = self.calculate_metrics(tp, tn, fp, fn)

        tp_v, tn_v, fp_v, fn_v = self.calculate_confusion_matrix(pred_valid, label_valid)
        acc_v, prec_v, sens_v, spec_v, f1_score_v = self.calculate_metrics(tp_v, tn_v, fp_v, fn_v)

        f = open(self.save_path + 'metrics.csv', 'a')
        f.write(str(step) + ";" + str(loss_total) + ";" + str(loss_valid) + ";" + str(tp) + ";" + str(tn) + ";" + str(
            fp) + ";" +
                str(fn) + ";" + str(acc) + ";" + str(prec) + ";" + str(sens) + ";" + str(spec) + ";" + str(f1_score) +
                ";" + str(tp_v) + ";" + str(tn_v) + ";" + str(fp_v) + ";" + str(fn_v) + ";" + str(acc_v) + ";" +
                str(prec_v) + ";" + str(sens_v) + ";" + str(spec_v) + ";" + str(f1_score_v) + "\n")
        f.close()

    def metrics_test(self, pred_image, label_image):
        tp, tn, fp, fn = self.calculate_confusion_matrix(pred_image, label_image)
        acc, prec, sens, spec, f1_score = self.calculate_metrics(tp, tn, fp, fn)

        f = open(self.save_path + 'metrics.csv', 'a')
        f.write(str(tp) + ";" + str(tn) + ";" + str(fp) + ";" + str(fn) + ";" + str(acc) + ";" + str(prec) + ";" +
                str(sens) + ";" + str(spec) + ";" + str(f1_score))
        f.close()


def prepare_pred_label(pred, label):
    """
        Help-function to flatten a prediction array and its
        corresponding label array.
        ----------
        pred : array, contains the precition of the processed
            images
        label : array, contains the labels of said images
        ----------
        y_pred : array, flattened pred array
        y_label : array, flattened label array
    """
    prep_label = label[:, :, :]
    prep_label = prep_label.flatten()
    prep_label = np.array(prep_label, dtype=bool)
    y_label = prep_label[:]

    prep_pred = pred[:, :, :]
    prep_pred = prep_pred.flatten()
    y_pred = prep_pred[:]

    return y_pred, y_label


def precision_recall(pred, label, step, path_save, model_path, f1_threshold):
    """
        Creates the precision recall curve of the given
        predictions and their corresponding labels.
        ----------
        pred : array, contains the precition of the processed
            images
        label : array, contains the labels of said images
        step : int, current epoch
        path_save : string, path where the curve is saved
    """
    y_pred, y_label = prepare_pred_label(pred, label)

    precision, recall, thresholds = precision_recall_curve(y_label, y_pred)
    calc_auc = auc(recall, precision)

    f = open(model_path + 'thresholds.csv', 'a')
    f.write("Thresholds;F1Score;Precision;Recall\n")
    for i in range(0, thresholds.shape[0]):
        f1_sc = 2 * ((precision[i] * recall[i]) / (precision[i] + recall[i]))
        if f1_sc > f1_threshold:
            f.write(str(thresholds[i]) + ";" + str(f1_sc) + ";" + str(precision[i]) + ";" +
                    str(recall[i]) + "\n")
    f.close()

    plt.plot([1, 0], [0, 1], linestyle='--')
    plt.plot(recall, precision, label='PR (Area = {:.3f})'.format(calc_auc))
    plt.xlabel('Recall (TPR)')
    plt.ylabel('Precision (PPV)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(path_save + 'metrics/pr_%d.tif' % step, dpi=400)
    plt.clf()


def roc(pred, label, step, path_save):
    """
        Creates the ROC of the given predictions and their
        corresponding labels.
        ----------
        pred : array, contains the precition of the processed
            images
        label : array, contains the labels of said images
        step : int, current epoch
        path_save : string, path where the curve is saved
    """
    y_pred, y_label = prepare_pred_label(pred, label)

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    calc_auc = auc(fpr, tpr)

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, label='PR (Area = {:.3f})'.format(calc_auc))
    plt.xlabel('(FPR)')
    plt.ylabel('(TPR)')
    plt.title('ROC')
    plt.legend(loc='best')
    plt.savefig(path_save + 'metrics/roc_%d.tif' % step, dpi=400)
    plt.clf()


def create_metrics_from_csv(path_csv):
    """
        Creates the loss, accuracy and F1-Score plot from
        the generated csv-file.
        ----------
        path_csv: string, path to the folder in which the
        metrics folder is stored
    """
    path = path_csv + "metrics/"
    csv = np.genfromtxt(path + 'metrics.csv', delimiter=";")
    epoch = csv[1:, 0]
    loss_train = csv[1:, 1]
    loss_valid = csv[1:, 2]
    acc_train = csv[1:, 7]
    acc_valid = csv[1:, 15]
    f1_train = csv[1:, 11]
    f1_valid = csv[1:, 20]

    x_min = 0
    x_max = int(epoch[len(loss_train) - 1])
    y_loss_min = -0.01
    y_loss_max = 0.2

    y_acc_min = -0.01
    y_acc_max = 1

    y_f1_min = -0.01
    y_f1_max = 1

    axes = plt.gca()
    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_loss_min, y_loss_max])

    plt.plot(epoch, loss_train, label='Loss Train')
    plt.plot(epoch, loss_valid, label='Loss Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend(loc='best')
    plt.savefig(path + 'loss.tif', dpi=400)
    plt.clf()

    plt.ylim(y_acc_min, y_acc_max)
    plt.xlim(x_min, x_max)

    plt.plot(epoch, acc_train, label='Accuracy Train')
    plt.plot(epoch, acc_valid, label='Accuracy Valid')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend(loc='best')
    plt.savefig(path + 'accuracy.tif', dpi=400)
    plt.clf()

    plt.ylim(y_f1_min, y_f1_max)
    plt.xlim(x_min, x_max)

    plt.plot(epoch, f1_train, label='F1 Score Train')
    plt.plot(epoch, f1_valid, label='F1 Score Valid')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend(loc='best')
    plt.savefig(path + 'f1.tif', dpi=400)
    plt.clf()


def show_array_info(name, array):
    print(name + ' dtype:', array.dtype)
    print(name + ' shape:' + str(np.shape(array)))
    print(name + ' range:', str(np.min(array)) + ' to ' + str(np.max(array)))
