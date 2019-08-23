import tensorflow as tf


def gradients_wrapper(y, x, dim_y, dim_x, stop_gradients=None):
    """

    :param y: (batch_size, dim_y)
    :param x: (batch_size, dim_x)
    :param dim_y:
    :param dim_x:
    :return: (batch_size, dim_y, dim_x)
    """
    jac_array = []
    for i in range(dim_y):
        jac, = tf.gradients(ys=y[:, i], xs=[x], stop_gradients=stop_gradients)  # FIXME: stop_gradients?
        jac_array.append(jac)

    return tf.stack(jac_array, axis=1)  # FIXME: is it safe not to separate envs?

