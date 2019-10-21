import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import ipdb as pdb


def my_accuracy(labels, logits):
    # act_labels = tf.argmax(labels, axis=-1, output_type=tf.dtypes.int32)
    # pred_labels = tf.argmax(logits, axis=-1, output_type=tf.dtypes.int32)
    act_labels = tf.argmax(labels)
    pred_labels = tf.argmax(logits)
    acc = tf.reduce_sum(tf.to_float(tf.equal(act_labels, pred_labels))) / tf.to_float(tf.shape(act_labels)[0])
    return acc


class ProtoNet(tf.keras.Model):

    def __init__(self, num_filters, latent_dim):
        super(ProtoNet, self).__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        num_filter_list = self.num_filters + [latent_dim]
        self.convs = []
        for i, num_filter in enumerate(num_filter_list):
            block_parts = [
                layers.Conv2D(
                    filters=num_filter,
                    kernel_size=3,
                    padding='SAME',
                    activation='linear'),
            ]

            block_parts += [layers.BatchNormalization()]
            block_parts += [layers.Activation('relu')]
            block_parts += [layers.MaxPool2D()]
            block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
            self.__setattr__("conv%d" % i, block)
            self.convs.append(block)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inp):
        out = inp
        for conv in self.convs:
            out = conv(out)
        out = self.flatten(out)
        return out


def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):
    """
        calculates the prototype network loss using the latent representation of x
        and the latent representation of the query set
        Args:
            x_latent: latent representation of supports with shape [N*S, D], where D is the latent dimension
            q_latent: latent representation of queries with shape [N*Q, D], where D is the latent dimension
            labels_onehot: one-hot encodings of the labels of the queries with shape [N, Q, N]
            num_classes: number of classes (N) for classification
            num_support: number of examples (S) in the support set
            num_queries: number of examples (Q) in the query set
        Returns:
            ce_loss: the cross entropy loss between the predicted labels and true labels
            acc: the accuracy of classification on the queries
    """
    #############################
    #### YOUR CODE GOES HERE ####
    pdb.set_trace()

    # compute the prototypes
    ck = tf.reduce_mean(tf.reshape(x_latent, [num_classes, num_support, -1]), axis=1)

    # compute the distance from the prototypes
    ck_norm = tf.nn.l2_normalize(ck, axis=1)
    q_latent_norm = tf.nn.l2_normalize(q_latent, axis=1)
    proto_cos_dists = 1 - tf.matmul(q_latent_norm, ck_norm, transpose_b=True)
    proto_logits = -1 * proto_cos_dists
    labels_re = tf.reshape(labels_onehot, [-1, num_classes])

    # compute cross entropy loss
    ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels_re, logits=proto_logits))
    acc = tf.contrib.metrics.accuracy(predictions=tf.argmax(proto_logits, 1), labels=tf.argmax(labels_re, 1))
    # note - additional steps are needed!

    # return the cross-entropy loss and accuracy
    # ce_loss, acc = None, None
    #############################
    return ce_loss, acc
