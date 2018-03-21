import math
import tensorflow as tf

from ad import conn


def weights_var(n1, n2, k):
    """
    Returns a TensorFlow variable for weights.

    https://www.tensorflow.org/api_docs/python/tf/truncated_normal
    """
    assert k == 1

    # TODO: Use
    # tf.truncated_normal([n1, n2, k], stddev=1.0 / math.sqrt(float(n1))),
    w = tf.Variable(
        tf.truncated_normal([n1, n2], stddev=1.0 / math.sqrt(float(n1))),
        name='w')
    tf.summary.histogram("weights", w, collections=['eval'])
    return w


def biases_var(size):
    """Returns a TensorFlow variable for biases."""
    b = tf.Variable(tf.constant(0.1, shape=[size]), name='b')
    tf.summary.histogram("biases", b, collections=['eval'])
    return b


def activation_histogram(v):
    tf.summary.histogram("activations", v, collections=['eval'])


def prediction_op_1(x, x_size, k):
    """
    Returns a prediction based on a neural network (NN).

    The NN will have this architecture:
    0. Input layer    (size: x_size)
    1. Hidden layer 1 (size: h1_size)
    2. Hidden layer 2 (size: h2_size)
    3. Hidden layer 3 (size: h3_size)
    4. Output layer   (size: x_size)

    Arguments:
    * `x` is the input vector

    Returns a TensorFlow variable that will hold the predicted
    value when the model is run.
    """
    assert k == 1

    # Build connection matrices
    cm12 = conn.matrix(x_size, 1, 2)
    cm23 = conn.matrix(x_size, 2, 3)
    cm34 = conn.matrix(x_size, 3, 4)

    h1_size = cm12.shape[1]
    h2_size = cm23.shape[1]
    h3_size = cm34.shape[1]

    with tf.name_scope('h1'):
        w = weights_var(x_size, h1_size, k)
        h1 = tf.nn.relu(
            tf.matmul(x, w * cm12) +
            biases_var(h1_size))
        activation_histogram(h1)

    with tf.name_scope('h2'):
        w = weights_var(h1_size, h2_size, k)
        h2 = tf.nn.relu(
            tf.matmul(h1, w * cm23) +
            biases_var(h2_size))
        activation_histogram(h2)

    with tf.name_scope('h3'):
        w = weights_var(h2_size, h3_size, k)
        h3 = tf.nn.relu(
            tf.matmul(h2, w * cm34) +
            biases_var(h3_size))
        activation_histogram(h3)

    with tf.name_scope('out'):
        w = weights_var(h3_size, x_size, k)
        out = tf.matmul(h3, w) + biases_var(x_size)
        activation_histogram(out)
        return out


def prediction_op_2(x, x_size, k):
    """
    Returns a prediction based on a neural network (NN).

    The NN will have this architecture:
    0. Input layer    (size: x_size)
    1. Hidden layer 1 (size: see code)
    2. Hidden layer 2 (size: see code)
    3. Hidden layer 3 (size: see code)
    4. Output layer   (size: x_size)

    Arguments:
    * `x` is the input vector

    Returns a TensorFlow variable that will hold the predicted
    value when the model is run.
    """
    assert k == 1

    # Build connection matrices
    cm01 = conn.matrix(x_size, 1, 2)
    cm02 = conn.matrix(x_size, 1, 3)
    cm12 = conn.matrix(x_size, 2, 3)
    cm23 = conn.matrix(x_size, 3, 4)

    h1_size = cm01.shape[1]
    h2_size = cm12.shape[1]
    h3_size = cm23.shape[1]

    with tf.name_scope('h1'):
        w_01 = weights_var(x_size, h1_size, k)
        h1 = tf.nn.relu(
            tf.matmul(x, w_01 * cm01) +
            biases_var(h1_size))
        activation_histogram(h1)

    with tf.name_scope('h2'):
        w_02 = weights_var(x_size, h2_size, k)
        w_12 = weights_var(h1_size, h2_size, k)
        h2 = tf.nn.relu(
            tf.matmul(x, w_02 * cm02) +
            tf.matmul(h1, w_12 * cm12) +
            biases_var(h2_size))
        activation_histogram(h2)

    with tf.name_scope('h3'):
        w_23 = weights_var(h2_size, h3_size, k)
        h3 = tf.nn.relu(
            tf.matmul(h2, w_23 * cm23) +
            biases_var(h3_size))
        activation_histogram(h3)

    with tf.name_scope('out'):
        w_3x = weights_var(h3_size, x_size, k)
        out = tf.matmul(h3, w_3x) + biases_var(x_size)
        activation_histogram(out)
        return out


def prediction_op_3(x, x_size, k):
    """
    Returns a prediction based on a neural network (NN).

    The NN will have this architecture:
    0. Input layer    (size: x_size)
    1. Hidden layer 1 (size: see code)
    2. Hidden layer 2 (size: see code)
    3. Hidden layer 3 (size: see code)
    4. Output layer   (size: x_size)

    Arguments:
    * `x` is the input vector

    Returns a TensorFlow variable that will hold the predicted
    value when the model is run.
    """
    assert k == 1

    # Build connection matrices
    cm01 = conn.matrix(x_size, 1, 2)
    cm02 = conn.matrix(x_size, 1, 3)
    cm12 = conn.matrix(x_size, 2, 3)
    cm13 = conn.matrix(x_size, 2, 4)
    cm23 = conn.matrix(x_size, 3, 4)

    h1_size = cm01.shape[1]
    h2_size = cm12.shape[1]
    h3_size = cm23.shape[1]

    with tf.name_scope('h1'):
        w_01 = weights_var(x_size, h1_size, k)
        h1 = tf.nn.relu(
            tf.matmul(x, w_01 * cm01) +
            biases_var(h1_size))
        activation_histogram(h1)

    with tf.name_scope('h2'):
        w_02 = weights_var(x_size, h2_size, k)
        w_12 = weights_var(h1_size, h2_size, k)
        h2 = tf.nn.relu(
            tf.matmul(x, w_02 * cm02) +
            tf.matmul(h1, w_12 * cm12) +
            biases_var(h2_size))
        activation_histogram(h2)

    with tf.name_scope('h3'):
        w_13 = weights_var(h1_size, h3_size, k)
        w_23 = weights_var(h2_size, h3_size, k)
        h3 = tf.nn.relu(
            tf.matmul(h1, w_13 * cm13) +
            tf.matmul(h2, w_23 * cm23) +
            biases_var(h3_size))
        activation_histogram(h3)

    with tf.name_scope('out'):
        w_3x = weights_var(h3_size, x_size, k)
        out = tf.matmul(h3, w_3x) + biases_var(x_size)
        activation_histogram(out)
        return out


def prediction_op_4(x, x_size, k):
    """
    Returns a prediction based on a neural network (NN).

    The NN will have this architecture:
    0. Input layer    (size: x_size)
    1. Hidden layer 1 (size: h1_size)
    2. Hidden layer 2 (size: h2_size)
    3. Hidden layer 3 (size: h3_size)
    4. Hidden layer 4 (size: h4_size)
    4. Output layer   (size: x_size)

    Arguments:
    * `x` is the input vector

    Returns a TensorFlow variable that will hold the predicted
    value when the model is run.
    """
    assert k == 1

    # Build connection matrices
    cm01 = conn.matrix(x_size, 1, 2)

    cm02 = conn.matrix(x_size, 1, 3)
    cm12 = conn.matrix(x_size, 2, 3)

    cm03 = conn.matrix(x_size, 1, 4)
    cm13 = conn.matrix(x_size, 2, 4)
    cm23 = conn.matrix(x_size, 3, 4)

    cm14 = conn.matrix(x_size, 2, 5)
    cm24 = conn.matrix(x_size, 3, 5)
    cm34 = conn.matrix(x_size, 4, 5)

    h1_size = cm01.shape[1]
    h2_size = cm12.shape[1]
    h3_size = cm23.shape[1]
    h4_size = cm34.shape[1]

    with tf.name_scope('h1'):
        w_01 = weights_var(x_size, h1_size, k)
        h1 = tf.nn.relu(
            tf.matmul(x, w_01 * cm01) +
            biases_var(h1_size))
        activation_histogram(h1)

    with tf.name_scope('h2'):
        w_02 = weights_var(x_size, h2_size, k)
        w_12 = weights_var(h1_size, h2_size, k)
        h2 = tf.nn.relu(
            tf.matmul(x, w_02 * cm02) +
            tf.matmul(h1, w_12 * cm12) +
            biases_var(h2_size))
        activation_histogram(h2)

    with tf.name_scope('h3'):
        w_03 = weights_var(x_size, h3_size, k)
        w_13 = weights_var(h1_size, h3_size, k)
        w_23 = weights_var(h2_size, h3_size, k)
        h3 = tf.nn.relu(
            tf.matmul(x, w_03 * cm03) +
            tf.matmul(h1, w_13 * cm13) +
            tf.matmul(h2, w_23 * cm23) +
            biases_var(h3_size))
        activation_histogram(h3)

    with tf.name_scope('h4'):
        w_14 = weights_var(h1_size, h4_size, k)
        w_24 = weights_var(h2_size, h4_size, k)
        w_34 = weights_var(h3_size, h4_size, k)
        h4 = tf.nn.relu(
            tf.matmul(h1, w_14 * cm14) +
            tf.matmul(h2, w_24 * cm24) +
            tf.matmul(h3, w_34 * cm34) +
            biases_var(h4_size))
        activation_histogram(h4)

    with tf.name_scope('out'):
        weights_4x = weights_var(h4_size, x_size, k)
        out = tf.matmul(h4, weights_4x) + biases_var(x_size)
        activation_histogram(out)
        return out


def prediction_fn(model_id):
    """
    Returns a prediction function for the given id.
    """
    if model_id == 1:
        return prediction_op_1
    elif model_id == 2:
        return prediction_op_2
    elif model_id == 3:
        return prediction_op_3
    elif model_id == 4:
        return prediction_op_4
    else:
        raise (RuntimeError('invalid model_id:{}'.format(model_id)))
