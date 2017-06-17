from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *
from utils import *


class DCGAN(object):
    def __init__(self, sess, g_iters=250000, input_height=108, input_width=108, is_crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 z_dim=100, gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024,
                 c_dim=3, dataset_name='celebA', d_iter_times=5, input_fname_pattern='*.jpg',
                 clamp_lower=-0.01, clamp_upper=0.01, checkpoint_dir="checkpoint", sample_dir="sample",
                 learning_rate=0.00005, logfile="logfile", logs="logs"):
        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.logs = logs

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.learning_rate = learning_rate

        self.c_dim = c_dim
        self.g_iters = g_iters
        self.d_iters = d_iter_times
        self.clamp_lower = clamp_lower
        self.clamp_upper = clamp_upper

        self.threshold = 0.5
        self.d_penalty = 0.001
        self.g_penalty = 0.001
        self.logfile = open(logfile, "w")

    def build_model(self):
        if self.is_crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_height, self.c_dim]

        inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

        add_noise_indicator = tf.placeholder(tf.float32, name="add_noise_indicator")

        z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        z_sum = tf.summary.histogram("z", z)

        G = self.generator(z)
        D_logits = self.discriminator(inputs)
        D_logits_ = self.discriminator(G, reuse=True)
        G_sum = tf.summary.image("G", G)

        g_loss = tf.reduce_mean(tf.scalar_mul(-1, BRF(D_logits_, self.threshold, add_noise_indicator)))
        d_loss = tf.reduce_mean(tf.subtract(D_logits_, D_logits))

        d_real_loss_sum = tf.summary.histogram("d", D_logits)
        d_fake_loss_sum = tf.summary.histogram("d_", D_logits_)
        g_loss_sum = tf.summary.scalar("g_loss", g_loss)
        d_loss_sum = tf.summary.scalar("d_loss", d_loss)

        t_vars = tf.trainable_variables()

        d_vars = [var for var in t_vars if 'd_' in var.name]
        clipped_d_vars = [tf.assign(var, tf.clip_by_value(var, self.clamp_lower, self.clamp_upper)) for var in d_vars]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        d_l2norm = tf.reduce_sum(tf.concat(
            0,
            [tf.reshape(var_1, [-1]) for var_1 in [tf.square(var_0) for var_0 in d_vars]]
            # , 0
        ))

        g_l2norm = tf.reduce_sum(tf.concat(
            0,
            [tf.reshape(var_1, [-1]) for var_1 in [tf.square(var_0) for var_0 in g_vars]]
            # , 0
        ))

        d_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        g_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        g_optim = g_optimizer.minimize(g_loss
                                       + self.g_penalty * g_l2norm
                                       , var_list=g_vars)

        d_optim = d_optimizer.minimize(d_loss
                                       + self.d_penalty * d_l2norm
                                       , var_list=d_vars)

        with tf.control_dependencies([d_optim]):
            d_optim = tf.tuple(clipped_d_vars)

        g_sum = tf.summary.merge([z_sum, G_sum, d_fake_loss_sum, g_loss_sum])
        d_sum = tf.summary.merge([z_sum, d_real_loss_sum, d_loss_sum])

        return z, add_noise_indicator, inputs, g_loss, d_loss, g_optim, d_optim, g_sum, d_sum, D_logits, D_logits_

    def train(self):
        """Train DCGAN"""
        [z, add_noise_indicator, inputs, g_loss, d_loss, g_optim, d_optim,
         g_sum, d_sum, D_logits, D_logits_] = self.build_model()
        self.saver = tf.train.Saver()
        counter = 1
        data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
        batch_idxs = len(data) // self.batch_size
        idx = 0
        batch_images, batch_z, idx = self.get_next_batch_data(idx, batch_idxs, data)
        writer = tf.summary.FileWriter("./" + self.logs, self.sess.graph)
        sampler = self.sampler(z)

        tf.global_variables_initializer().run()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # set parameters for the training process
        start_time = time.time()

        for g_subiters in range(self.g_iters):

            if g_subiters < 25 or g_subiters % 500 == 0:
                d_iters = 100
            else:
                d_iters = self.d_iters

            # Update D network
            for j in range(d_iters):
                _, summary_str = self.sess.run([d_optim, d_sum], feed_dict={inputs: batch_images, z: batch_z})
                writer.add_summary(summary_str, counter)
                errD = d_loss.eval({z: batch_z, inputs: batch_images})
                dv_real = D_logits.eval({inputs: batch_images})
                dv_fake = D_logits_.eval({z: batch_z})
                self.logfile.write("g_iters: [%2d] discriminator training [%4d/%4d] time: %4.4f, d_loss: %.8f, avg_dv_real: %8f, avg_dv_fake: %8f\n" %
                                   (g_subiters, j+1, d_iters, time.time() - start_time, errD, dv_real.mean(), dv_fake.mean()))
                batch_images, batch_z, idx = self.get_next_batch_data(idx, batch_idxs, data)

            # Update G network
            for i in range(int(g_subiters * 25 / self.g_iters) + 1):
                random_num = np.random.random(1)[0]
                _, summary_str = \
                    self.sess.run([g_optim, g_sum], feed_dict={z: batch_z, add_noise_indicator: random_num})
                writer.add_summary(summary_str, counter)
                batch_images, batch_z, idx = self.get_next_batch_data(idx, batch_idxs, data)
                errD = d_loss.eval({z: batch_z, inputs: batch_images})
                errG = g_loss.eval({z: batch_z, add_noise_indicator: 2.0})
                dv_real = D_logits.eval({inputs: batch_images})
                dv_fake = D_logits_.eval({z: batch_z})
                self.logfile.write("g_iters: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, avg_dv_real: %8f, avg_dv_fake: %8f\n" %
                                   (g_subiters, i+1, int(g_subiters * 25 / self.g_iters) + 1,
                                    time.time() - start_time, errD, errG, dv_real.mean(), dv_fake.mean()))

            counter += 1
            batch_images, batch_z, idx = self.get_next_batch_data(idx, batch_idxs, data)
            if np.mod(counter, 100) == 1:
                sample_inputs, sample_z, idx = self.get_next_batch_data(idx, batch_idxs, data)
                samples, sample_d_loss, sample_g_loss = self.sess.run([sampler, d_loss, g_loss],
                                                                      feed_dict={z: sample_z, inputs: sample_inputs,
                                                                                 add_noise_indicator: 2.0})
                save_images(samples, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, g_subiters, idx))
                self.logfile.write("[Sample] d_loss: %.8f, g_loss: %.8f\n" % (sample_d_loss, sample_g_loss))

            if np.mod(counter, 500) == 2:
                self.save(self.checkpoint_dir, counter)

    def get_next_batch_data(self, idx, batch_idxs, data):
        if idx == batch_idxs:
            np.random.shuffle(data)
            idx = 0
        batch_files = data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch = [get_image(batch_file,
                           input_height=self.input_height,
                           input_width=self.input_width,
                           resize_height=self.output_height,
                           resize_width=self.output_width,
                           is_crop=self.is_crop) for batch_file in batch_files]

        batch_images = np.array(batch).astype(np.float32)
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        return batch_images, batch_z, idx+1

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(batch_norm(x=conv2d(h0, self.df_dim * 2, name='d_h1_conv'), name='d_bn1'))
            h2 = lrelu(batch_norm(x=conv2d(h1, self.df_dim * 4, name='d_h2_conv'), name='d_bn2'))
            h3 = lrelu(batch_norm(x=conv2d(h2, self.df_dim * 8, name='d_h3_conv'), name='d_bn3'))

            return tf.sigmoid(linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin'))

    def generator(self, z):
        with tf.variable_scope("generator"):
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = int(s_h / 2), int(s_h / 4), int(s_h / 8), int(s_h / 16)
            s_w2, s_w4, s_w8, s_w16 = int(s_w / 2), int(s_w / 4), int(s_w / 8), int(s_w / 16)

            # project `z` and reshape
            z_ = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin')
            h0 = tf.nn.relu(batch_norm(
                x=tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim * 8]),
                name='g_bn0'))

            h1 = tf.nn.relu(batch_norm(
                x=deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1'),
                name='g_bn1'))

            h2 = tf.nn.relu(batch_norm(
                x=deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2'),
                name='g_bn2'))

            h3 = tf.nn.relu(batch_norm(
                x=deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3'),
                name='g_bn3'))

            return tf.nn.tanh(deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4'))

    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = int(s_h / 2), int(s_h / 4), int(s_h / 8), int(s_h / 16)
            s_w2, s_w4, s_w8, s_w16 = int(s_w / 2), int(s_w / 4), int(s_w / 8), int(s_w / 16)

            # project `z` and reshape
            h0 = tf.reshape(linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'), [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(batch_norm(x=h0, train=False, name='g_bn0'))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
            h1 = tf.nn.relu(batch_norm(x=h1, train=False, name='g_bn1'))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
            h2 = tf.nn.relu(batch_norm(x=h2, train=False, name='g_bn2'))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
            h3 = tf.nn.relu(batch_norm(x=h3, train=False, name='g_bn3'))

            return tf.nn.tanh(deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4'))

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...\n")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}\n".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint\n")
            return False


def BRF(pred, threshold, noise_indicator):
    return tf.cond(
        tf.less(noise_indicator, tf.constant(threshold)),
        lambda: tf.subtract(tf.ones(pred.get_shape()), pred),
        lambda: tf.identity(pred))
