"""
TensorFlow model.
"""

import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import time

from ad import gen, VERSION
import ad.models as models


def generate_synthetic_data(n, dimensions):
    d = dimensions
    b = 10
    g = 5
    print('Generating synthetic data...')
    return gen.observations(n, d, b, g)


def make_dataset(data_placeholder):
    """
    Returns a Tensorflow Dataset.

    See also:
    * https://www.tensorflow.org/programmers_guide/datasets
    * https://www.tensorflow.org/versions/master/get_started/datasets_quickstart
    """
    global flags
    dataset = tf.data.Dataset.from_tensor_slices(data_placeholder)
    dataset = dataset.shuffle(
        buffer_size=flags.shuffle_buffer_size,
        seed=flags.seed_shuffle)
    dataset = dataset.batch(flags.batch_size)
    return dataset


def training_loss_op(labels, predictions):
    """
    List of TensorFlow loss functions:
    https://www.tensorflow.org/api_docs/python/tf/losses
    """
    with tf.name_scope('loss'):
        mse = tf.losses.mean_squared_error(
            labels=labels, predictions=predictions)
        tf.summary.scalar('mse', mse, collections=['train'])
        return mse


def evaluation_op(labels, predictions):
    """
    Returns a tuple with two components:

    * mean_squared_error: A Tensor representing the current mean, the value of
      total divided by count.
    * update_op: An operation that increments the total and count variables
      appropriately and whose value matches mean_squared_error.

    See:
    https://www.tensorflow.org/api_docs/python/tf/metrics/mean_squared_error
    """
    with tf.name_scope('eval'):
        mse, update = tf.metrics.mean_squared_error(
            labels=labels, predictions=predictions)
        tf.summary.scalar('mse', mse, collections=['eval'])
        return mse, update


def training_op(loss, learning_rate):
    """
    Returns the training operation.

    * Creates a summarizer to track the loss over time in TensorBoard.
    * Creates an optimizer and applies the gradients to all trainable
      variables.

    Returns a TensorFlow Op (operation). To start training, pass this
    return value to `sess.run()`.

    See also:
    * tf.train.GradientDescentOptimizer
    * tf.train.AdamOptimizer
    * tf.train.AdagradOptimizer

    Source:
    https://www.tensorflow.org/api_guides/python/train
    """
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


def hparam_str():
    """Returns hyperparameter-based name."""
    global flags
    return 'id={},dim={},nm={},lr={},b={}'.format(
        flags.model_id,
        flags.dimensions,
        flags.node_multiplier,
        flags.learning_rate,
        flags.batch_size)


def train_and_evaluate_model():
    """
    Train and evaluate the model.
    """
    global flags
    x_size = flags.dimensions
    train_data = generate_synthetic_data(flags.train_data_size, x_size)
    test_data = generate_synthetic_data(flags.test_data_size, x_size)

    with tf.Graph().as_default() as graph:
        if flags.seed_tf:
            tf.set_random_seed(flags.seed_tf)
        with tf.name_scope('data'):
            data_pl = tf.placeholder(train_data.dtype, name='data')
            dataset = make_dataset(data_pl)
            data_it = dataset.make_initializable_iterator()
            x = data_it.get_next()

        prediction_fn = models.prediction_fn(flags.model_id)
        y = prediction_fn(x, x_size, flags.node_multiplier)

        loss_op = training_loss_op(x, y)
        train_op = training_op(loss_op, flags.learning_rate)

        train_summary_op = tf.summary.merge_all(key='train')

        mse_op, update_op = evaluation_op(x, y)
        eval_summary_op = tf.summary.merge_all(key='eval')

        init = [
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ]

        model_dir = os.path.join(flags.base_dir, hparam_str())
        print(flags.base_dir)
        print(model_dir)
        summary_writer = tf.summary.FileWriter(model_dir, graph=graph)

        # Start training session
        with tf.Session() as sess:
            step = 0

            def eval_model():
                evaluate_model({
                    'data_iterator': data_it,
                    'data_placeholder': data_pl,
                    'data_source': test_data,
                    'epoch': epoch,
                    'mse_op': mse_op,
                    'session': sess,
                    'step': step,
                    'summary_op': eval_summary_op,
                    'summary_writer': summary_writer,
                    'update_op': update_op,
                })

            sess.run(init)
            # Run a fixed number of training epochs
            for epoch in range(flags.train_epochs):
                step = train_model({
                    'data_iterator': data_it,
                    'data_placeholder': data_pl,
                    'data_source': train_data,
                    'epoch': epoch,
                    'loss_op': loss_op,
                    'session': sess,
                    'step': step,
                    'summary_op': train_summary_op,
                    'summary_writer': summary_writer,
                    'train_op': train_op,
                })
                # Evaluate every `flags.eval_epochs` epochs.
                if epoch % flags.eval_epochs == 0:
                    eval_model()

            # Final evaluation
            eval_model()

        summary_writer.close()


def train_model(args: dict):
    """Trains model for one epoch."""
    global flags
    sess = args['session']
    sess.run(args['data_iterator'].initializer, feed_dict={
        args['data_placeholder']: args['data_source']})

    i = 0
    step = args['step']
    start_time = None
    summary_str = None

    def summarize():
        nonlocal start_time
        duration = time.time() - start_time
        start_time = None
        args['summary_writer'].add_summary(summary_str, step)
        print("[T] epoch:{:3d} i:{:5d} dt:{:.4f} step:{:6d} loss:{:.6f}"
              .format(args['epoch'], i, duration, step, loss_value))

    while True:
        try:
            if not start_time:
                start_time = time.time()
            _, loss_value, summary_str = sess.run([
                args['train_op'], args['loss_op'], args['summary_op']])
            if i > 0 and i % flags.loss_steps == 0:
                summarize()
            step += 1
            i += 1
        except tf.errors.OutOfRangeError:
            # No more data is available from the data iterator
            if start_time:
                summarize()
            break

    return step


def evaluate_model(args: dict):
    sess = args['session']
    sess.run(args['data_iterator'].initializer, feed_dict={
        args['data_placeholder']: args['data_source']})
    step = args['step']
    start_time = time.time()
    mse_val = None
    summary_str = None
    while True:
        try:
            mse_val, update_val, summary_str = sess.run(
                [args['mse_op'], args['update_op'], args['summary_op']])
        except tf.errors.OutOfRangeError:
            break
    duration = time.time() - start_time
    args['summary_writer'].add_summary(summary_str, step)
    print("[E] epoch:{:3d}         dt:{:.4f} step:{:6d}                "
          "MSE:{:.6f}".format(args['epoch'], duration, step, mse_val))


def build_cli_parser():
    parser = argparse.ArgumentParser()

    # Model architecture params
    parser.add_argument(
        '--model_id', type=int, default=None,
        help='identifies which model to use')
    parser.add_argument(
        '--node_multiplier', type=int, default=1,
        help='number of nodes per combination')

    # Data params
    parser.add_argument(
        '--dimensions', type=int, default=10,
        help='number of dimensions in an observation')
    parser.add_argument(
        '--seed_np', type=int, default=None,
        help='random number seed for NumPy')
    parser.add_argument(
        '--seed_shuffle', type=int, default=None,
        help='random number seed for TF shuffling')
    parser.add_argument(
        '--seed_tf', type=int, default=None,
        help='random number seed for TensorFlow')
    parser.add_argument(
        '--shuffle_buffer_size', type=int, default=1000,
        help='size of shuffle buffer')

    # Training params
    parser.add_argument(
        '--batch_size', type=int, default=100,
        help='number of examples per batch')
    parser.add_argument(
        '--loss_steps', type=int, default=1000,
        help='number of steps between loss display')
    parser.add_argument(
        '--train_data_size', type=int, default=20000,
        help='size of synthetic training data')
    parser.add_argument(
        '--train_epochs', type=int, default=30,
        help='number of training epochs')

    # Testing params
    parser.add_argument(
        '--test_data_size', type=int, default=10000,
        help='size of synthetic testing data')
    parser.add_argument(
        '--eval_epochs', type=int, default=5,
        help='evaluate after this number of epochs')

    # Optimization params
    parser.add_argument(
        '--learning_rate', type=float, default=0.1,
        help='number of examples per batch')

    # Directory params
    parser.add_argument(
        '--base_dir', type=str, default='/tmp/ad1_model',
        help='base directory for the model')
    return parser


flags = {}


def print_flags():
    global flags
    print("- base_dir            : {}".format(flags.base_dir))
    print("- batch_size          : {}".format(flags.batch_size))
    print("- dimensions          : {}".format(flags.dimensions))
    print("- eval_epochs         : {}".format(flags.eval_epochs))
    print("- learning_rate       : {}".format(flags.learning_rate))
    print("- loss_steps          : {}".format(flags.loss_steps))
    print("- model_id            : {}".format(flags.model_id))
    print("- node_multiplier     : {}".format(flags.node_multiplier))
    print("- seed_np             : {}".format(flags.seed_np))
    print("- seed_shuffle        : {}".format(flags.seed_shuffle))
    print("- seed_tf             : {}".format(flags.seed_tf))
    print("- shuffle_buffer_size : {}".format(flags.shuffle_buffer_size))
    print("- test_data_size      : {}".format(flags.test_data_size))
    print("- train_data_size     : {}".format(flags.train_data_size))
    print("- train_epochs        : {}".format(flags.train_epochs))


def main(_):
    global flags
    if flags.seed_np:
        np.random.seed(seed=flags.seed_np)
    train_and_evaluate_model()


def cli_start():
    global flags
    print("Anomaly Detector, version={}".format(VERSION))
    tf.logging.set_verbosity(tf.logging.INFO)
    cli_parser = build_cli_parser()
    flags, unparsed = cli_parser.parse_known_args()
    print_flags()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    cli_start()
