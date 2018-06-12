import functools

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import official.mnist.dataset as mnist
from official.utils.flags import core as flags_core
import numpy as np
from absl import flags, app as absl_app
import os.path as opth
import os


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


def discriminator_model():
    conv_kwargs = dict(kernel_size=5, padding='same', activation=None)
    model = models.Sequential(name="Discriminator")

    suffix = "Discriminator"

    # Leaky ReLU with alpa=0.2
    activation_fn = functools.partial(tf.nn.leaky_relu, alpha=0.2)

    def conv2d_dropout(filters, strides, activation=activation_fn, name_suffix=suffix, index=0):
        suffix = name_suffix + str(index)
        model.add(layers.Conv2D(
            filters=filters, strides=strides, **conv_kwargs, name="Conv{}".format(suffix)))
        model.add(layers.Dropout(name="Dropout{}".format(suffix), rate=0.3))
        model.add(layers.Activation(activation=activation, name="Activation{}".format(suffix)))

    # Three blocks of convs and dropouts
    conv2d_dropout(filters=32, strides=2, index=0)
    conv2d_dropout(filters=64, strides=2, index=1)
    conv2d_dropout(filters=64, strides=1, index=2)

    # Flatten it and build logit layer
    model.add(layers.Flatten(name="Flatten{}".format(suffix)))
    model.add(layers.Dense(units=1, activation=None, name="Out{}".format(suffix)))

    return model


def define_flags():
    flags_core.define_base()  # Defines batch_size and train_epochs
    flags_core.define_image()
    flags_core.set_defaults(batch_size=32, train_epochs=15)
    flags_core.flags.DEFINE_float(name="lr", default=2e-4, help="Learning rate")


def main(_):
    flags_obj = flags.FLAGS
    with tf.Graph().as_default():
        images = prepare_input_pipeline(flags_obj)

        # Set up the models
        with tf.name_scope("Discriminator") as d_scope:
            d_model = discriminator_model()
        with tf.name_scope("Generator") as g_scope:
            g_model = generator_model()

        # Build the models
        with tf.name_scope("LatentNoiseVector"):
            z = tfd.Normal(loc=0.0, scale=1.0).sample(sample_shape=(tf.shape(images)[0], 100))
        with tf.name_scope(g_scope):
            generated_images = g_model(z)
        with tf.name_scope(d_scope):
            with tf.name_scope("Real"):
                d_real = d_model(images)
            with tf.name_scope("Fake") as discriminate_fake_scope:
                d_fake = d_model(generated_images)

        # Set the discriminator losses
        with tf.name_scope("DiscriminatorLoss"):
            loss_d_real = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(d_real) * 0.9, logits=d_real)
            loss_d_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(d_real), logits=d_fake)
            loss_d = tf.reduce_mean(tf.concat([loss_d_real, loss_d_fake], axis=0))

        # Configure discriminator training ops
        with tf.name_scope("Train") as train_scope:
            optimizer = tf.train.AdamOptimizer(flags_obj.lr, beta1=0.5)
            optimize_d = optimizer.minimize(loss_d, var_list=d_model.trainable_variables)
        with tf.name_scope(discriminate_fake_scope):
            with tf.control_dependencies([optimize_d]):
                # Build a second time, so that new variables are used
                d_fake = d_model(generated_images)

        # Set the generator loss and the actual train op
        with tf.name_scope("GeneratorLoss"):
            loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(d_fake), logits=d_fake))
        with tf.name_scope(train_scope):
            train_op = optimizer.minimize(loss_g, var_list=g_model.trainable_variables)

        # Setup summaries
        with tf.name_scope("Summaries"):
            summary_op = tf.summary.merge([
                tf.summary.scalar("LossDiscriminator", loss_d),
                tf.summary.scalar("LossGenerator", loss_g),
                tf.summary.image("GeneratedImages", generated_images)])
        writer = tf.summary.FileWriter(_next_logdir("tensorboard"))
        writer.add_graph(tf.get_default_graph())

        # Run training
        steps_per_epoch = 50_000 // flags_obj.batch_size
        print(g_model.summary())
        print(d_model.summary())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            for epoch in range(flags_obj.train_epochs):
                losses_d, losses_g = [], []
                print("Epoch {}".format(epoch))
                for _ in range(steps_per_epoch):
                    if step % 100 == 0:
                        _, loss_g_out, loss_d_out, summ = sess.run(
                            [train_op, loss_g, loss_d, summary_op])  # Look Ma, no feed_dict!
                        writer.add_summary(summ, global_step=step)
                    else:
                        _, loss_g_out, loss_d_out = sess.run([train_op, loss_g, loss_d])
                    losses_d.append(loss_d_out)
                    losses_g.append(loss_g_out)
                    step += 1

                print("Discriminator loss: {0:.4f}, Generator loss: {1:.4f}".format(
                    np.mean(losses_d), np.mean(losses_g)))


def prepare_input_pipeline(flags_obj):
    with tf.name_scope("InputPipeline"):
        ds = mnist.train("/home/jos/datasets/mnist")
        ds = ds.cache() \
            .shuffle(buffer_size=50000) \
            .batch(flags_obj.batch_size) \
            .repeat()\
            .make_one_shot_iterator()
        images, _ = ds.get_next()
        # Reshape and rescale to [-1, 1]
        return tf.reshape(images, (-1, 28, 28, 1)) * 2.0 - 1.0


def _next_logdir(path):
    os.makedirs(path, exist_ok=True)
    subdirs = [d for d in os.listdir(path) if opth.isdir(opth.join(path, d))]
    if len(subdirs) == 0:
        runid = 0
    else:
        runid = len(subdirs)
    logdir = opth.join(path, "run" + str(runid).zfill(4))
    os.makedirs(logdir)
    return logdir


if __name__ == "__main__":
    define_flags()
    absl_app.run(main)
