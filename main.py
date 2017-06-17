import os
# from model import DCGAN
from model_BRF import DCGAN
import pprint
import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_integer("g_iters", 250000, "g_iters default to [25000]")
flags.DEFINE_float("learning_rate", 0.00005, "Learning rate of for RProp [0.00005]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_BRF_nt0.5", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples_BRF_nt0.5", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [True]")
flags.DEFINE_string("logfile", "logfile_BRF_nt0.5", "some output for debugging")
flags.DEFINE_string("logs", "logs_BRF_nt0.5", "some output for debugging")
FLAGS = flags.FLAGS


def main(_):
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    np.random.seed(11)
    tf.set_random_seed(1234)

    with tf.Session(config=run_config) as sess:

        dcgan = DCGAN(
            sess,
            g_iters=FLAGS.g_iters,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            c_dim=FLAGS.c_dim,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            is_crop=FLAGS.is_crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            logfile=FLAGS.logfile,
            logs=FLAGS.logs)

        dcgan.train()

if __name__ == '__main__':
    tf.app.run()
