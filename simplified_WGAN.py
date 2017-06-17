from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
from glob import glob
import tensorflow.contrib.layers as ly
from utils import get_image

batch_size = 64
z_dim = 128
learning_rate_ger = 5e-5
learning_rate_dis = 5e-5

# img size
original_size = 108
size = 64
# update Citers times of critic in one iter(unless i < 25 or i % 500 == 0, i is iterstep)
Citers = 5
# the upper bound and lower bound of parameters in critic
clamp_lower = -0.01
clamp_upper = 0.01

channel = 3

# directory to store log, including loss and grad_norm of generator and critic
log_dir = './log_wgan'
ckpt_dir = './ckpt_wgan'
dataset_name = 'celebA'
input_fname_pattern = '*.jpg'
logfile_name = "logfile-simp"

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
# max iter step, note the one step indicates that a Citers updates of critic and one update of generator
max_iter_step = 20000


def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def generator_conv(z):
    with tf.variable_scope('generator'):
        train = ly.fully_connected(
            z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=ly.batch_norm)

        train = tf.reshape(train, (-1, 4, 4, 512))

        train = ly.conv2d_transpose(train, 256, 3, stride=2,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME',
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))

        train = ly.conv2d_transpose(train, 128, 3, stride=2,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME',
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))

        train = ly.conv2d_transpose(train, 64, 3, stride=2,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME',
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))

        train = ly.conv2d_transpose(train, channel, 3, stride=2,
                                    activation_fn=tf.nn.tanh, padding='SAME',
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
    return train


def critic_conv(img, reuse=False):
    with tf.variable_scope('critic') as scope:
        if reuse:
            scope.reuse_variables()

        img = ly.conv2d(img, num_outputs=size, kernel_size=3, stride=2, activation_fn=lrelu)

        img = ly.conv2d(img, num_outputs=size * 2, kernel_size=3, stride=2, activation_fn=lrelu,
                        normalizer_fn=ly.batch_norm)

        img = ly.conv2d(img, num_outputs=size * 4, kernel_size=3, stride=2, activation_fn=lrelu,
                        normalizer_fn=ly.batch_norm)

        img = ly.conv2d(img, num_outputs=size * 8, kernel_size=3, stride=2, activation_fn=lrelu,
                        normalizer_fn=ly.batch_norm)

        logit = ly.fully_connected(tf.reshape(img, [batch_size, -1]), 1, activation_fn=None)

    return logit


# def build_graph():
#     z = tf.placeholder(tf.float32, shape=(batch_size, z_dim))
#     generator = generator_conv
#     critic = critic_conv
#     with tf.variable_scope('generator'):
#         train = generator(z)
#     real_data = tf.placeholder(dtype=tf.float32, shape=(batch_size, size, size, channel))
#     true_logit = critic(real_data)
#     fake_logit = critic(train, reuse=True)
#     c_loss = tf.reduce_mean(fake_logit - true_logit)
#     g_loss = tf.reduce_mean(-fake_logit)
#
#     g_loss_sum = tf.summary.scalar("g_loss", g_loss)
#     c_loss_sum = tf.summary.scalar("c_loss", c_loss)
#     img_sum = tf.summary.image("img", train, max_outputs=10)
#
#     theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
#
#     theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
#
#     counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
#
#     opt_g = ly.optimize_loss(loss=g_loss, learning_rate=learning_rate_ger,
#                              optimizer=tf.train.RMSPropOptimizer,
#                              variables=theta_g, global_step=counter_g,
#                              summaries='gradient_norm')
#
#     counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
#
#     opt_c = ly.optimize_loss(loss=c_loss, learning_rate=learning_rate_dis,
#                              optimizer=tf.train.RMSPropOptimizer,
#                              variables=theta_c, global_step=counter_c,
#                              summaries='gradient_norm')
#
#     clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_c]
#     # merge the clip operations on critic variables
#     with tf.control_dependencies([opt_c]):
#         opt_c = tf.tuple(clipped_var_c)
#     return opt_g, opt_c, z, real_data, c_loss, g_loss


def main():
    data = glob(os.path.join("./data", dataset_name, input_fname_pattern))
    batch_idxs = len(data) // batch_size
    idx = 0

    logfile = open(logfile_name, "w")

    # opt_g, opt_c, z, real_data, c_loss, g_loss = build_graph()
    z = tf.placeholder(tf.float32, shape=(batch_size, z_dim))
    train = generator_conv(z)
    real_data = tf.placeholder(dtype=tf.float32, shape=(batch_size, size, size, channel))
    true_logit = critic_conv(real_data)
    fake_logit = critic_conv(train, reuse=True)
    c_loss = tf.reduce_mean(fake_logit - true_logit)
    g_loss = tf.reduce_mean(-fake_logit)

    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    c_loss_sum = tf.summary.scalar("c_loss", c_loss)
    img_sum = tf.summary.image("img", train, max_outputs=10)

    theta_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    theta_c = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

    counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

    opt_g = ly.optimize_loss(loss=g_loss, learning_rate=learning_rate_ger,
                             optimizer=tf.train.RMSPropOptimizer,
                             variables=theta_g, global_step=counter_g,
                             summaries='gradient_norm')

    counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

    opt_c = ly.optimize_loss(loss=c_loss, learning_rate=learning_rate_dis,
                             optimizer=tf.train.RMSPropOptimizer,
                             variables=theta_c, global_step=counter_c,
                             summaries='gradient_norm')

    clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_c]
    # merge the clip operations on critic variables
    with tf.control_dependencies([opt_c]):
        opt_c = tf.tuple(clipped_var_c)

    merged_all = tf.summary.merge_all()

    saver = tf.train.Saver()

    def next_feed_dict(idx_in, batch_idxs_in, data_in):
        if idx_in == batch_idxs_in:
            np.random.shuffle(data_in)
            idx_in = 0
        batch_files = data_in[idx_in * batch_size:(idx_in + 1) * batch_size]
        batch = [get_image(batch_file,
                           input_height=original_size,
                           input_width=original_size,
                           resize_height=size,
                           resize_width=size,
                           is_crop=True) for batch_file in batch_files]

        train_img = np.array(batch).astype(np.float32)
        if train_img.shape != (64, 64, 64, 3):
            print(batch, idx_in, batch_idxs_in)
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
        feed_dict_in = {real_data: train_img, z: batch_z}
        return feed_dict_in, idx_in + 1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        for i in range(max_iter_step):
            if i < 25 or i % 500 == 0:
                citers = 100
            else:
                citers = Citers
            for j in range(citers):
                feed_dict, idx = next_feed_dict(idx, batch_idxs, data)
                if i % 100 == 99 and j == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, merged = sess.run([opt_c, merged_all], feed_dict=feed_dict,
                                         options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(run_metadata, 'critic_metadata {}'.format(i), i)
                else:
                    sess.run(opt_c, feed_dict=feed_dict)

                errC = c_loss.eval(feed_dict=feed_dict)
                print("g_iters: [%2d] discriminator training [%4d/%4d], d_loss: %.8f\n" %
                      (i, j, citers, errC))
                logfile.write("g_iters: [%2d] discriminator training [%4d/%4d], d_loss: %.8f\n" %
                              (i, j, citers, errC))

            if i % 100 == 99:
                _, merged = sess.run([opt_g, merged_all],
                                     # feed_dict={z: np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)},
                                     feed_dict=feed_dict,
                                     options=run_options, run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.add_run_metadata(run_metadata, 'generator_metadata {}'.format(i), i)

                # feed_dict, idx = next_feed_dict(idx, batch_idxs, data)
                # samples, c_loss, g_loss = sess.run([self.sampler, self.d_loss, self.g_loss],
                #                                         feed_dict={self.z: sample_z, self.inputs: sample_inputs})
                # # save_images(samples, [8, 8],
                # #             './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, g_iters, self.idx))
                # self.logfile.write("[Sample] d_loss: %.8f, g_loss: %.8f\n" % (d_loss, g_loss))
            else:
                sess.run(opt_g,
                         # feed_dict={z: np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)},
                         feed_dict=feed_dict,
                         )
                errG = g_loss.eval(
                    # {z: np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)}
                    feed_dict
                )
                print ("g_iters: [%2d] [%4d/%4d], d_loss: %.8f, g_loss: %.8f\n" %
                       (i, idx, batch_idxs, errC, errG))
                logfile.write("g_iters: [%2d] [%4d/%4d], d_loss: %.8f, g_loss: %.8f\n" %
                              (i, idx, batch_idxs, errC, errG))
            if i % 1000 == 999:
                saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=i)

if __name__ == '__main__':
    main()