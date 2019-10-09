import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers
import ipdb as pdb
import pandas as pd
from collections import defaultdict

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 4,
                     'Number of N-way classification tasks per batch')

flags.DEFINE_integer('nl', 1,
                     'Number of lstm layers (excluding the last layer)')

flags.DEFINE_float('lr', 0.0001, 'learning rate')

flags.DEFINE_integer('ns', 500000, 'Number of steps')

flags.DEFINE_integer('hs', 128, 'lstm hidden size')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    #############################
    #### YOUR CODE GOES HERE ####
    # pdb.set_trace()
    last_preds = preds[:, -1, :, :]
    last_labels = labels[:, -1, :, :]
    N = last_preds.get_shape().as_list()[-1]
    last_preds_re = tf.reshape(last_preds, [-1, N])
    last_labels_re = tf.reshape(last_labels, [-1, N])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=last_labels_re, logits=last_preds_re)
    # loss = tf.losses.softmax_cross_entropy(labels[:, -1, :, :], preds[:, -1, :, :])
    return loss
    #############################


class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer_pipeline = [tf.keras.layers.LSTM(FLAGS.hs, return_sequences=True) for _ in range(FLAGS.nl)]
        print(self.layer_pipeline)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        # pdb.set_trace()
        # label_list = tf.unstack(input_labels, axis=1)
        # last_labs = tf.zeros_like(label_list[-1])
        # new_image_labels = tf.stack(label_list[:-1] + [last_labs], axis=1)
        # concat_inputs = tf.concat([input_images, new_image_labels], axis=-1)

        # zeros_mask = tf.zeros_like(input_labels[:, -1:, :, :])  # [B, 1, N, N]
        # masked_input_labels = tf.concat([input_labels[:, :-1, :, :], zeros_mask], axis=1)
        # concat_inputs = tf.concat([input_images, masked_input_labels], axis=-1)

        concat_inputs = tf.concat([input_images, input_labels], axis=-1)
        _, K_1, N, D = concat_inputs.get_shape().as_list()
        o1 = tf.reshape(concat_inputs, [-1, K_1 * N, D])
        # o1 = self.layer1(concat_inputs_reshaped)
        # o1 = tf.identity(concat_inputs_reshaped)
        for layer in self.layer_pipeline:
            o1 = layer(o1)
        o2 = self.layer2(o1)
        # o2 = self.layer2(concat_inputs_reshaped)
        out = tf.reshape(o2, [-1, K_1, N, N])

        #############################
        return out


ims = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
labels = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))
labels_mask = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))
# pdb.set_trace()
data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1)
# test = data_generator.sample_batch("train", 10)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels_mask)
# pdb.set_trace()

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(FLAGS.lr)
optimizer_step = optim.minimize(loss)
ddict = defaultdict(list)
outfile = "N_{}_K_{}_B_{}_lr_{}_NS_{}_HS_{}_NL_{}.csv".format(FLAGS.num_classes, FLAGS.num_samples,
                                                              FLAGS.meta_batch_size,
                                                              FLAGS.lr, FLAGS.ns, FLAGS.hs, FLAGS.nl)

try:
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for step in range(FLAGS.ns):
            i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
            # pdb.set_trace()
            l_mask = l.copy()
            l_mask[:, -1, :, :] = 0.0
            feed = {ims: i.astype(np.float32),
                    labels: l.astype(np.float32),
                    labels_mask: l_mask.astype(np.float32),
                    }
            _, ls = sess.run([optimizer_step, loss], feed)

            if step % 100 == 0:
                print("*" * 5 + "Iter " + str(step) + "*" * 5)
                i, l = data_generator.sample_batch('test', 100)
                l_mask = l.copy()
                l_mask[:, -1, :, :] = 0.0
                feed = {ims: i.astype(np.float32),
                        labels: l.astype(np.float32),
                        labels_mask: l_mask.astype(np.float32),
                        }
                pred, tls = sess.run([out, loss], feed)
                print("Train Loss:", ls, "Test Loss:", tls)
                ddict['train_loss'].append(ls)
                ddict['test_loss'].append(tls)
                pred = pred.reshape(
                    -1, FLAGS.num_samples + 1,
                    FLAGS.num_classes, FLAGS.num_classes)
                pred = pred[:, -1, :, :].argmax(2)
                l = l[:, -1, :, :].argmax(2)
                print("Test Accuracy", (1.0 * (pred == l)).mean())
                ddict['test_acc'].append((1.0 * (pred == l)).mean())
finally:
    df = pd.DataFrame(ddict)
    df.to_csv(outfile)
# pdb.set_trace()
