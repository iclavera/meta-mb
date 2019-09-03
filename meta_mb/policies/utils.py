import tensorflow as tf


def jacobian_wrapper(y, x, dim_y, dim_x=None, stop_gradients=None):
    """

    :param y: (batch_size, dim_y)
    :param x: (batch_size, dim_x)
    :param dim_y:
    :param dim_x:
    :return: (batch_size, dim_y, dim_x)
    """
    if isinstance(x, list):
        assert y.get_shape().ndims == 1
        jac_array_by_dim = []  # list with length dim_y
        for i in range(dim_y):
            jac_i_array = tf.gradients(ys=y[i], xs=x, stop_gradients=stop_gradients)
            jac_array_by_dim.append(jac_i_array)
        jac_array = list(zip(*jac_array_by_dim))
        return list(map(lambda jac_i: tf.stack(jac_i, axis=0), jac_array))

    if y.get_shape().ndims == 1:
        jac_array = []
        for i in range(dim_y):
            jac, = tf.gradients(ys=y[i], xs=[x], stop_gradients=stop_gradients)
            assert jac is not None
            jac_array.append(jac)
        return tf.stack(jac_array, axis=0)

    if y.get_shape().ndims == 2:
        jac_array = []
        for i in range(dim_y):
            jac, = tf.gradients(ys=y[:, i], xs=[x], stop_gradients=stop_gradients)  # FIXME: stop_gradients?
            assert jac is not None
            jac_array.append(jac)
        return tf.stack(jac_array, axis=1)  # FIXME: is it safe not to separate envs?

def hessian_wrapper(y, x, dim_y, dim_x=None):
    """

    :param y: (batch_size, dim_y)
    :param x: (batch_size, dim_x)
    :param dim_y:
    :param dim_x:
    :param stop_gradient:
    :return:
    """
    assert y.get_shape().ndims == 1
    hess_array = []
    for i in range(dim_y):
        hess, = tf.hessians(ys=y[i], xs=[x])  # (dim_x, dim_x)
        assert hess is not None
        hess_array.append(hess)
        assert hess.get_shape().as_list() == [dim_x, dim_x]
    return tf.stack(hess_array, axis=0)  # (dim_y, dim_x, dim_x)
