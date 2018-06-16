import functools

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from official.utils.flags import core as flags_core
import numpy as np
from absl import flags, app as absl_app
import os.path as opth
import os
from sklearn.utils import shuffle


layers = tf.keras.layers
models = tf.keras.models


def generator_model():
    model = models.Sequential(name="Generator")
    conv_kwargs = dict(kernel_size=5, padding='same', activation=None)

    def conv2d_block(filters, strides, transpose=True, activation=tf.nn.relu,
                         suffix="Generator", index=0):
        suffix = suffix + str(index)
        if transpose:
            model.add(layers.UpSampling2D(name="Upsampling" + suffix))
            model.add(layers.Conv2D(
                filters=filters, **conv_kwargs, name="Conv2DTranspose" + suffix))
        else:
            model.add(layers.Conv2D(filters=filters, strides=strides, **conv_kwargs,
                                    name="Conv2D" + suffix))
        model.add(layers.Activation(activation=activation, name="Activation" + suffix))

    model.add(layers.Dense(7 * 7 * 64, activation=tf.nn.relu))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((7, 7, 64)))

    conv2d_block(filters=128, strides=2, transpose=True, index=0)
    conv2d_block(filters=64, strides=2, transpose=True, index=1)
    conv2d_block(filters=64, strides=1, transpose=False, index=2)
    conv2d_block(filters=1, strides=1, transpose=False, activation=tf.nn.tanh, index=3)
    return model


class Discriminator:

    def __init__(self):
        self.tail = self._define_tail()
        self.head = self._define_head()

    def _define_tail(self, name="Discriminator"):
        conv_kwargs = dict(kernel_size=5, padding='same', activation=None)
        feature_model = models.Sequential(name=name)

        # Leaky ReLU with alpa=0.2
        activation_fn = functools.partial(tf.nn.leaky_relu, alpha=0.2)

        def conv2d_dropout(filters, strides, activation=activation_fn, index=0):
            suffix = str(index)
            feature_model.add(layers.Conv2D(
                filters=filters, strides=strides, **conv_kwargs, name="Conv{}".format(suffix)))
            feature_model.add(layers.Dropout(name="Dropout{}".format(suffix), rate=0.3))
            feature_model.add(layers.Activation(
                activation=activation, name="Activation{}".format(suffix)))

        # Three blocks of convs and dropouts
        conv2d_dropout(filters=32, strides=2, index=0)
        conv2d_dropout(filters=64, strides=2, index=1)
        conv2d_dropout(filters=64, strides=1, index=2)

        # Flatten it and build logits layer
        feature_model.add(layers.Flatten(name="Flatten"))
        return feature_model

    def _define_head(self):
        head_model = models.Sequential(name="DiscriminatorHead")
        head_model.add(layers.Dense(units=10, activation=None, name="Out"))
        return head_model

    @property
    def trainable_variables(self):
        return self.tail.trainable_variables + self.head.trainable_variables

    def __call__(self, x, *args, **kwargs):
        features = self.tail(x, *args, **kwargs)
        return self.head(features, *args, **kwargs), features


def define_flags():
    flags_core.define_base()  # Defines batch_size and train_epochs
    flags_core.define_image()
    flags_core.set_defaults(batch_size=32, train_epochs=15)
    flags_core.flags.DEFINE_float(name="lr", default=1e-2, help="Learning rate")
    flags_core.flags.DEFINE_float(name="stddev", default=2e-4, help="Learning rate")
    flags_core.flags.DEFINE_integer(name="num_classes", default=10, help="Number of classes")
    flags_core.flags.DEFINE_integer(name="z_dim_size", default=100,
                                    help="Dimension of noise vector")
    flags_core.flags.DEFINE_integer(name="num_labeled_examples", default=400,
                                    help="Number of labeled examples per class")


def main(_):
    flags_obj = flags.FLAGS
    with tf.Graph().as_default():
        (images_lab, labels_lab), (images_unl, labels_unl), (images_test, labels_test) = \
            prepare_input_pipeline(flags_obj)

        # Setup noise vector
        with tf.name_scope("LatentNoiseVector"):
            z = tfd.Normal(loc=0.0, scale=flags_obj.stddev).sample(
                sample_shape=(tf.shape(images_lab)[0], flags_obj.z_dim_size))

        # Generate images from noise vector
        with tf.name_scope("Generator"):
            g_model = generator_model()
            generated_images = g_model(z)

        # Discriminate between real and fake, and try to classify the labeled data
        with tf.name_scope("Discriminator") as discriminator_scope:
            d_model = Discriminator()
            logits_fake, features_fake = d_model(generated_images)
            logits_real_unl, features_real_unl = d_model(images_unl)
            logits_real_lab, features_real_lab = d_model(images_lab)

        # Set the discriminator losses
        with tf.name_scope("DiscriminatorLoss"):
            # Supervised loss, just cross-entropy. This normalizes p(y|x) where 1 <= y <= K
            loss_supervised = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_lab, logits=logits_real_lab)

            # Sum of unnormalized log probabilities
            logits_sum_real = tf.reduce_logsumexp(logits_real_unl, axis=1)
            logits_sum_fake = tf.reduce_logsumexp(logits_fake, axis=1)
            loss_unsupervised = 0.5 * (
                tf.negative(tf.reduce_mean(logits_sum_real)) +
                tf.reduce_mean(tf.nn.softplus(logits_sum_real)) +
                tf.reduce_mean(tf.nn.softplus(logits_sum_fake)))
            loss_d = loss_supervised + loss_unsupervised

        def accuracy(logits, labels):
            preds = tf.argmax(logits, axis=1)
            return tf.reduce_mean(tf.to_float(tf.equal(preds, labels)))

        # Configure discriminator training ops
        with tf.name_scope("Train") as train_scope:
            optimizer = tf.train.AdamOptimizer(flags_obj.lr, beta1=0.5)
            optimize_d = optimizer.minimize(loss_d, var_list=d_model.trainable_variables)
            train_accuracy_op = 0.5 * (
                    accuracy(logits_real_lab, labels_lab) + accuracy(logits_real_unl, labels_unl))

        with tf.name_scope(discriminator_scope):
            with tf.control_dependencies([optimize_d]):
                # Build a second time, so that new variables are used
                logits_fake, features_fake = d_model(generated_images, training=True)
                logits_real_unl, features_real_unl = d_model(images_unl, training=True)

        # Set the generator loss and the actual train op
        with tf.name_scope("GeneratorLoss"):
            feature_mean_real = tf.reduce_mean(features_real_unl, axis=0)
            feature_mean_fake = tf.reduce_mean(features_fake, axis=0)
            # L1 distance of features is the loss for the generator
            loss_g = tf.reduce_mean(tf.abs(feature_mean_real - feature_mean_fake))

        with tf.name_scope(train_scope):
            train_op = optimizer.minimize(loss_g, var_list=g_model.trainable_variables)

        with tf.name_scope(discriminator_scope):
            with tf.name_scope("Test"):
                test_accuracy_op = accuracy(d_model(images_test, training=False), labels_test)

        # Setup summaries
        with tf.name_scope("Summaries"):
            summary_op = tf.summary.merge([
                tf.summary.scalar("LossDiscriminator", loss_d),
                tf.summary.scalar("LossGenerator", loss_g),
                tf.summary.image("GeneratedImages", generated_images)])
        writer = tf.summary.FileWriter(_next_logdir("tensorboard/mnist_ssl"))
        writer.add_graph(tf.get_default_graph())

        # Run training
        steps_per_epoch = 50_000 // flags_obj.batch_size
        steps_per_test = 10_000 // flags_obj.batch_size
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            for epoch in range(flags_obj.train_epochs):
                losses_d, losses_g, accuracies = [], [], []
                print("Epoch {}".format(epoch))
                for _ in range(steps_per_epoch):
                    if step % 100 == 0:
                        # Look Ma, no feed_dict!
                        _, loss_g_batch, loss_d_batch, summ, accuracy_batch = sess.run(
                            [train_op, loss_g, loss_d, summary_op, train_accuracy_op])  
                        writer.add_summary(summ, global_step=step)
                    else:
                        _, loss_g_batch, loss_d_batch, accuracy_batch = sess.run(
                            [train_op, loss_g, loss_d, train_accuracy_op])
                    losses_d.append(loss_d_batch)
                    losses_g.append(loss_g_batch)
                    accuracies.append(accuracy_batch)
                    step += 1

                print("Discriminator loss: {0:.4f}, Generator loss: {1:.4f}, "
                      "Train accuracy: {2:.4f}"
                      .format(np.mean(losses_d), np.mean(losses_g), np.mean(accuracies)))

                # Classify test data
                accuracies = [sess.run(test_accuracy_op) for _ in range(steps_per_test)]
                print("Test accuracy: {0:.4f}", np.mean(accuracies))


def prepare_input_pipeline(flags_obj):
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data(
        "/home/jos/datasets/mnist")

    def reshape_and_scale(x, img_shape=(-1, 28, 28, 1)):
        return x.reshape(img_shape) * 2.0 - 1.0

    # Reshape data and rescale to [-1, 1]
    train_x = reshape_and_scale(train_x)
    test_x = reshape_and_scale(test_x)

    # Shuffle train data
    train_x_unlabeled, train_y_unlabeled = shuffle(train_x, train_y)

    # Select subset as supervised
    train_x_labeled, train_y_labeled = [], []
    for i in range(flags_obj.num_classes):
        train_x_labeled.append(
            train_x_unlabeled[train_y_unlabeled == i][:flags_obj.num_labeled_examples])
        train_y_labeled.append(
            train_y_unlabeled[train_y_unlabeled == i][:flags_obj.num_labeled_examples])
    train_x_labeled = np.concatenate(train_x_labeled)
    train_y_labeled = np.concatenate(train_y_labeled)

    with tf.name_scope("InputPipeline"):

        def train_pipeline(data, shuffle_buffer_size):
            return tf.data.Dataset.from_tensor_slices(data)\
                .cache()\
                .shuffle(buffer_size=shuffle_buffer_size)\
                .batch(flags_obj.batch_size)\
                .repeat()\
                .make_one_shot_iterator()

        # Setup pipeline for labeled data
        train_ds_lab = train_pipeline(
            (train_x_labeled, train_y_labeled),
            flags_obj.num_labeled_examples * flags_obj.num_classes)
        images_lab, labels_lab = train_ds_lab.get_next()

        # Setup pipeline for unlabeled data
        train_ds_unl = train_pipeline(
            (train_x_unlabeled, train_y_unlabeled), len(train_x_labeled))
        images_unl, labels_unl = train_ds_unl.get_next()

        # Setup pipeline for test data
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))\
            .cache()\
            .batch(flags_obj.batch_size)\
            .repeat()\
            .make_one_shot_iterator()
        images_test, labels_test = test_ds.get_next()

    return (images_lab, labels_lab), (images_unl, labels_unl), (images_test, labels_test)


def _next_logdir(path):
    os.makedirs(path, exist_ok=True)
    subdirs = [d for d in os.listdir(path) if opth.isdir(opth.join(path, d))]
    logdir = opth.join(path, "run" + str(len(subdirs)).zfill(4))
    os.makedirs(logdir)
    return logdir


if __name__ == "__main__":
    define_flags()
    absl_app.run(main)
