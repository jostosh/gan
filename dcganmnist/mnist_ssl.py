import tensorflow as tf
import tensorflow.contrib.distributions as tfd
from official.utils.flags import core as flags_core
import numpy as np
from absl import flags, app as absl_app
import os.path as opth
import tqdm
import os
from sklearn.utils import shuffle


layers = tf.keras.layers


def define_flags():
    flags_core.define_base()  # Defines batch_size and train_epochs
    flags_core.define_image()
    flags_core.set_defaults(
        batch_size=32, train_epochs=100)
    flags_core.flags.DEFINE_float(
        name="lr", default=2e-4,
        help="Learning rate")
    flags_core.flags.DEFINE_float(
        name="stddev", default=1e-2,
        help="z standard deviation")
    flags_core.flags.DEFINE_integer(
        name="num_classes", default=10,
        help="Number of classes")
    flags_core.flags.DEFINE_integer(
        name="z_dim_size", default=100,
        help="Dimension of noise vector")
    flags_core.flags.DEFINE_integer(
        name="num_labeled_examples", default=400,
        help="Number of labeled examples per class")
    flags_core.flags.DEFINE_bool(
        name="man_reg", default=False,
        help="Manifold regularization")


def define_generator():

    def conv2d_block(filters, upsample=True, activation=tf.nn.relu, index=0):
        if upsample:
            model.add(layers.UpSampling2D(name="UpSampling" + str(index), size=(2, 2)))
        model.add(layers.Conv2D(
            filters=filters, kernel_size=5, padding='same', name="Conv2D" + str(index),
            activation=activation))

    # From flat noise to spatial
    model = tf.keras.models.Sequential(name="Generator")
    model.add(layers.Dense(7 * 7 * 64, activation=tf.nn.relu, name="NoiseToSpatial"))
    model.add(layers.Reshape((7, 7, 64)))

    # Four blocks of convolutions, 2 that upsample and convolve, and 2 more that
    # just convolve
    conv2d_block(filters=128, upsample=True, index=0)
    conv2d_block(filters=64, upsample=True, index=1)
    conv2d_block(filters=64, upsample=False, index=2)
    conv2d_block(filters=1, upsample=False, activation=tf.nn.tanh, index=3)
    return model


class Discriminator:

    def __init__(self):
        """The discriminator network. Split up in a 'tail' and 'head' network, so that we can
        easily get the """
        self.tail = self._define_tail()
        self.head = self._define_head()

    def _define_tail(self, name="Discriminator"):
        """Defines the network until the intermediate layer that can be used for feature-matching
        loss."""
        feature_model = tf.keras.models.Sequential(name=name)

        def conv2d_dropout(filters, strides, index=0):
            # Adds a convolution followed by a Dropout layer
            suffix = str(index)
            feature_model.add(layers.Conv2D(
                filters=filters, strides=strides, name="Conv{}".format(suffix), padding='same',
                kernel_size=5, activation=tf.nn.leaky_relu))
            feature_model.add(layers.Dropout(name="Dropout{}".format(suffix), rate=0.3))

        # Three blocks of convs and dropouts. They all have 5x5 kernels, leaky ReLU and 0.3
        # dropout rate.
        conv2d_dropout(filters=32, strides=2, index=0)
        conv2d_dropout(filters=64, strides=2, index=1)
        conv2d_dropout(filters=64, strides=1, index=2)

        # Flatten it and build logits layer
        feature_model.add(layers.Flatten(name="Flatten"))
        return feature_model

    def _define_head(self):
        # Defines the remaining layers after the 'tail'
        head_model = tf.keras.models.Sequential(name="DiscriminatorHead")
        head_model.add(layers.Dense(units=10, activation=None, name="Logits"))
        return head_model

    @property
    def trainable_variables(self):
        # Return both tail's parameters a well as those of the head
        return self.tail.trainable_variables + self.head.trainable_variables

    def __call__(self, x, *args, **kwargs):
        # By adding this, the code below can treat a Discriminator instance as a
        # tf.keras.models.Sequential instance
        features = self.tail(x, *args, **kwargs)
        return self.head(features, *args, **kwargs), features


def accuracy(logits, labels):
    """Compute accuracy for this mini-batch """
    preds = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.to_float(tf.equal(preds, labels)))


def main(_):
    flags_obj = flags.FLAGS
    with tf.Graph().as_default():
        # Setup th einput pipeline
        (images_lab, labels_lab), (images_unl, labels_unl), (images_unl2, labels_unl2), \
            (images_test, labels_test) = prepare_input_pipeline(flags_obj)

        with tf.name_scope("BatchSize"):
            batch_size_tensor = tf.shape(images_lab)[0]

        # Get the noise vectors
        z, z_perturbed = define_noise(batch_size_tensor, flags_obj)

        # Generate images from noise vector
        with tf.name_scope("Generator"):
            g_model = define_generator()
            images_fake = g_model(z)
            images_fake_perturbed = g_model(z_perturbed)

        # Discriminate between real and fake, and try to classify the labeled data
        with tf.name_scope("Discriminator") as discriminator_scope:
            d_model = Discriminator()
            logits_fake, features_fake          = d_model(images_fake, training=True)
            logits_fake_perturbed, _            = d_model(images_fake_perturbed, training=True)
            logits_real_unl, features_real_unl  = d_model(images_unl, training=True)
            logits_real_lab, features_real_lab  = d_model(images_lab, training=True)
            logits_train, _                     = d_model(images_lab, training=False)

        # Set the discriminator losses
        with tf.name_scope("DiscriminatorLoss"):
            # Supervised loss, just cross-entropy. This normalizes p(y|x) where 1 <= y <= K
            loss_supervised = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_lab, logits=logits_real_lab))

            # Sum of unnormalized log probabilities
            logits_sum_real = tf.reduce_logsumexp(logits_real_unl, axis=1)
            logits_sum_fake = tf.reduce_logsumexp(logits_fake, axis=1)
            loss_unsupervised = 0.5 * (
                tf.negative(tf.reduce_mean(logits_sum_real)) +
                tf.reduce_mean(tf.nn.softplus(logits_sum_real)) +
                tf.reduce_mean(tf.nn.softplus(logits_sum_fake)))
            loss_d = loss_supervised + loss_unsupervised
            if flags_obj.man_reg:
                loss_d += 1e-3 * tf.nn.l2_loss(logits_fake - logits_fake_perturbed) \
                    / tf.to_float(batch_size_tensor)

        # Configure discriminator training ops
        with tf.name_scope("Train") as train_scope:
            optimizer = tf.train.AdamOptimizer(flags_obj.lr * 0.25)
            optimize_d = optimizer.minimize(loss_d, var_list=d_model.trainable_variables)
            train_accuracy_op = accuracy(logits_train, labels_lab)

        with tf.name_scope(discriminator_scope):
            with tf.control_dependencies([optimize_d]):
                # Build a second time, so that new variables are used
                logits_fake, features_fake = d_model(images_fake, training=True)
                logits_real_unl, features_real_unl = d_model(images_unl2, training=True)

        # Set the generator loss and the actual train op
        with tf.name_scope("GeneratorLoss"):
            feature_mean_real = tf.reduce_mean(features_real_unl, axis=0)
            feature_mean_fake = tf.reduce_mean(features_fake, axis=0)
            # L1 distance of features is the loss for the generator
            loss_g = tf.reduce_mean(tf.abs(feature_mean_real - feature_mean_fake))

        with tf.name_scope(train_scope):
            optimizer = tf.train.AdamOptimizer(flags_obj.lr, beta1=0.5)
            train_op = optimizer.minimize(loss_g, var_list=g_model.trainable_variables)

        with tf.name_scope(discriminator_scope):
            with tf.name_scope("Test"):
                logits_test, _ = d_model(images_test, training=False)
                test_accuracy_op = accuracy(logits_test, labels_test)

        # Setup summaries
        with tf.name_scope("Summaries"):
            summary_op = tf.summary.merge([
                tf.summary.scalar("LossDiscriminator", loss_d),
                tf.summary.scalar("LossGenerator", loss_g),
                tf.summary.image("GeneratedImages", images_fake),
                tf.summary.scalar("ClassificationAccuracyTrain", train_accuracy_op),
                tf.summary.scalar("ClassificationAccuracyTest", test_accuracy_op)])
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
                pbar = tqdm.trange(steps_per_epoch)
                for _ in pbar:
                    if step % 1000 == 0:
                        # Look Ma, no feed_dict!
                        _, loss_g_batch, loss_d_batch, summ, accuracy_batch = sess.run(
                            [train_op, loss_g, loss_d, summary_op, train_accuracy_op])  
                        writer.add_summary(summ, global_step=step)
                    else:
                        _, loss_g_batch, loss_d_batch, accuracy_batch = sess.run(
                            [train_op, loss_g, loss_d, train_accuracy_op])
                    pbar.set_description("Discriminator loss {0:.3f}, Generator loss {1:.3f}"
                                         .format(loss_d_batch, loss_g_batch))
                    losses_d.append(loss_d_batch)
                    losses_g.append(loss_g_batch)
                    accuracies.append(accuracy_batch)
                    step += 1

                print("Discriminator loss: {0:.4f}, Generator loss: {1:.4f}, "
                      "Train accuracy: {2:.4f}"
                      .format(np.mean(losses_d), np.mean(losses_g), np.mean(accuracies)))

                # Classify test data
                accuracies = [sess.run(test_accuracy_op) for _ in range(steps_per_test)]
                print("Test accuracy: {0:.4f}".format(np.mean(accuracies)))


def define_noise(batch_size_tensor, flags_obj):
    # Setup noise vector
    with tf.name_scope("LatentNoiseVector"):
        z = tfd.Normal(loc=0.0, scale=flags_obj.stddev).sample(
            sample_shape=(batch_size_tensor, flags_obj.z_dim_size))
        z_perturbed = z + tfd.Normal(loc=0.0, scale=flags_obj.stddev).sample(
            sample_shape=(batch_size_tensor, flags_obj.z_dim_size)) * 1e-5
    return z, z_perturbed


def prepare_input_pipeline(flags_obj):
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data(
        "/home/jos/datasets/mnist/mnist.npz")

    def reshape_and_scale(x, img_shape=(-1, 28, 28, 1)):
        return x.reshape(img_shape).astype(np.float32) / 255. * 2.0 - 1.0

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
            (train_x_labeled, train_y_labeled.astype(np.int64)),
            flags_obj.num_labeled_examples * flags_obj.num_classes)
        images_lab, labels_lab = train_ds_lab.get_next()

        # Setup pipeline for unlabeled data
        train_ds_unl = train_pipeline(
            (train_x_unlabeled, train_y_unlabeled.astype(np.int64)), len(train_x_labeled))
        images_unl, labels_unl = train_ds_unl.get_next()

        # Setup another pipeline that also uses the unlabeled data, so that we use a different
        # batch for computing the discriminator loss and the generator loss
        train_x_unlabeled, train_y_unlabeled = shuffle(train_x_unlabeled, train_y_unlabeled)
        train_ds_unl2 = train_pipeline(
            (train_x_unlabeled, train_y_unlabeled.astype(np.int64)), len(train_x_labeled))
        images_unl2, labels_unl2 = train_ds_unl2.get_next()

        # Setup pipeline for test data
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y.astype(np.int64)))\
            .cache()\
            .batch(flags_obj.batch_size)\
            .repeat()\
            .make_one_shot_iterator()
        images_test, labels_test = test_ds.get_next()

    return (images_lab, labels_lab), (images_unl, labels_unl), (images_unl2, labels_unl2), \
           (images_test, labels_test)


def _next_logdir(path):
    os.makedirs(path, exist_ok=True)
    subdirs = [d for d in os.listdir(path) if opth.isdir(opth.join(path, d))]
    logdir = opth.join(path, "run" + str(len(subdirs)).zfill(4))
    os.makedirs(logdir)
    return logdir


if __name__ == "__main__":
    define_flags()
    absl_app.run(main)
