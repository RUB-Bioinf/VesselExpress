# Weilin Fu (2019) Frangi-Net on High-Resolution Fundus (HRF) image database [Source Code].
# https://doi.org/10.24433/CO.5016803.v2
# GNU General Public License (GPL) */

import tensorflow as tf
from tensorflow.python.ops import array_ops


def focal_loss(prediction_tensor, target_tensor, alpha=0.25, gamma=2):
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    
    return per_entry_cross_ent


def get_loss(logits, labels, mask, weight):
    with tf.variable_scope('losses'):
        loss_wcr = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels,
                                                            pos_weight=10., name='weighted_cross_entropy')
        loss_fc = focal_loss(logits, labels, alpha=0.9, gamma=2.)

        loss_wcr = loss_wcr * weight
        loss_fc = loss_fc * weight

        if mask is not None:
            loss_wcr = tf.boolean_mask(loss_wcr, mask)
            loss_fc = tf.boolean_mask(loss_fc, mask)

        loss_wcr = tf.reduce_mean(loss_wcr)
        loss_fc = tf.reduce_mean(loss_fc)

        return loss_wcr, loss_fc

