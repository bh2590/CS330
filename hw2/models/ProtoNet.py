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


def calc_euc(inp):
    # pdb.set_trace()
    q, c = inp
    c = tf.expand_dims(c, axis=0)
    n = tf.norm(q - c, 'euclidean', -1)
    return [n]


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

    def euclidean_distance(a, b):
        # a.shape = N x D
        # b.shape = M x D
        N, D = tf.shape(a)[0], tf.shape(a)[1]
        M = tf.shape(b)[0]
        a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
        b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
        return tf.reduce_mean(tf.square(a - b), axis=2)

    x_prototypes = tf.reduce_mean(tf.reshape(x_latent, [num_classes, num_support, -1]), axis=1)

    q_latent_ti = tf.tile(tf.expand_dims(q_latent, axis=1), (1, tf.shape(x_prototypes)[0], 1))
    x_prototypes_ti = tf.tile(tf.expand_dims(x_prototypes, axis=0), (tf.shape(q_latent)[0], 1, 1))

    dists = tf.reduce_mean(tf.square(q_latent_ti - x_prototypes_ti), axis=2)
    log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])
    # log_p_y = tf.reshape(tf.nn.softmax(-dists), [num_classes, num_queries, num_classes])
    # ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels_onehot, logits=log_p_y))
    ce_loss = -tf.reduce_sum(tf.reshape(tf.reduce_sum(tf.multiply(labels_onehot, log_p_y), axis=-1), [-1]))
    labels_re = tf.reshape(labels_onehot, [-1, num_classes])
    proto_re = tf.reshape(log_p_y, [-1, num_classes])
    acc = tf.contrib.metrics.accuracy(predictions=tf.argmax(log_p_y, -1), labels=tf.argmax(labels_onehot, -1))

    # acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), tf.argmax(labels_onehot, axis=-1))))

    # # compute the prototypes
    # ck = tf.reduce_mean(tf.reshape(x_latent, [num_classes, num_support, -1]), axis=1)
    #
    # # compute the distance from the prototypes
    # q_latent2 = tf.reshape(tf.tile(q_latent, [num_classes, 1]), [num_classes, num_classes * num_queries, -1])
    # proto_euc_dists = tf.transpose(tf.map_fn(calc_euc, [q_latent2, ck], [tf.float32])[0])
    #
    # # ck_norm = tf.nn.l2_normalize(ck, axis=1)
    # # q_latent_norm = tf.nn.l2_normalize(q_latent, axis=1)
    # # proto_cos_dists = 1 - tf.matmul(q_latent_norm, ck_norm, transpose_b=True)
    # # proto_logits = -1 * proto_cos_dists
    # proto_logits = -1 * proto_euc_dists
    # labels_re = tf.reshape(labels_onehot, [-1, num_classes])
    #
    # # compute cross entropy loss
    # ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels_re, logits=proto_logits))
    # acc = tf.contrib.metrics.accuracy(predictions=tf.argmax(proto_logits, 1), labels=tf.argmax(labels_re, 1))
    # note - additional steps are needed!

    # return the cross-entropy loss and accuracy
    # ce_loss, acc = None, None
    #############################
    return ce_loss, acc
