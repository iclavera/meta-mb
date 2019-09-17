import tensorflow as tf
from collections import OrderedDict


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

    :param y: (dim_y)
    :param x: (dim_x)
    :param dim_y:
    :param dim_x:
    :param stop_gradient:
    :return:
    """
    assert y.get_shape().ndims == 1

    if isinstance(x, list):
        hess_array_by_dim = []
        for i in range(dim_y):
            hess_array_by_dim.append(tf.hessians(ys=y[i], xs=x))
        hess_array = list(zip(*hess_array_by_dim))
        return list(map(lambda hess_i: tf.stack(hess_i, axis=0), hess_array))

    hess_array = []
    for i in range(dim_y):
        hess, = tf.hessians(ys=y[i], xs=[x])  # (dim_x, dim_x)
        assert hess is not None
        hess_array.append(hess)
        assert hess.get_shape().as_list() == [dim_x, dim_x]
    return tf.stack(hess_array, axis=0)  # (dim_y, dim_x, dim_x)

def tf_batch_inner(a, b):
    """

    :param a: (N, B)
    :param b: (N, B)
    :return: (B,)
    """
    return tf.reduce_sum(tf.multiply(a, b), axis=0)

def tf_cg(f_Ax, b, cg_iters=20, residual_tol=1e-10):
    # if any column in the batch has residual >= tolerance, continue the loop
    cond = lambda p, r, x, rdotr: tf.greater_equal(tf.reduce_max(rdotr), residual_tol)
    def body(p, r, x, rdotr):
        z = f_Ax(p)
        # v = rdotr / tf.tensordot(p, z, axes=1)
        v = rdotr / tf_batch_inner(p, z)
        x += v*p
        r -= v*z
        # newrdotr = tf.tensordot(r, r, axes=1)
        newrdotr = tf_batch_inner(r, r)
        mu = newrdotr / rdotr
        p = r + mu*p
        return (p, r, x, newrdotr)

    p = tf.identity(b)
    r = tf.identity(b)
    x = tf.zeros_like(b)
    # rdotr = tf.tensordot(r, r, axes=1)
    rdotr = tf_batch_inner(r, r)
    loop_vars = (p, r, x, rdotr)

    p, r, x, rdotr = tf.while_loop(body=body, cond=cond, loop_vars=loop_vars, maximum_iterations=cg_iters)
    accept = tf.less(tf.reduce_max(rdotr), residual_tol)
    # x = tf.Print(x, data=['residual', rdotr])
    return accept, x

def flatten_jac_sym(jac_list, dim_y):
    jac_flatten_list = list(map(lambda jac: tf.reshape(jac, (dim_y, -1)), jac_list))
    jac_flatten = tf.concat(jac_flatten_list, axis=-1)
    return jac_flatten

def flatten_params_sym(params):
    return tf.concat([tf.reshape(param, (-1,)) for param in params.values()], axis=0)

def unflatten_params_sym(flatten_params, params_example):
    unflatten_params = []
    ptr = 0
    for key, param in params_example.items():
        flatten_size = tf.reduce_prod(tf.shape(param))
        param = tf.reshape(flatten_params[ptr:ptr+flatten_size], tf.shape(param))
        unflatten_params.append((key, param))
        ptr += flatten_size
    return OrderedDict(unflatten_params)

# def tf_inner(a, b):
#     return tf.reduce_sum(tf.multiply(a, b))
#
# def tf_cg(f_Ax, b, cg_iters=10, residual_tol=1e-10):
#     cond = lambda p, r, x, rdotr: tf.greater_equal(rdotr, residual_tol)
#     def body(p, r, x, rdotr):
#         z = f_Ax(p)
#         # v = rdotr / tf.tensordot(p, z, axes=1)
#         v = rdotr / tf_inner(p, z)
#         x += v*p
#         r -= v*z
#         # newrdotr = tf.tensordot(r, r, axes=1)
#         newrdotr = tf_inner(r, r)
#         mu = newrdotr / rdotr
#         p = r + mu*p
#         return (p, r, x, newrdotr)
#
#     p = tf.identity(b)
#     r = tf.identity(b)
#     x = tf.zeros_like(b)
#     # rdotr = tf.tensordot(r, r, axes=1)
#     rdotr = tf_inner(r, r)
#     loop_vars = (p, r, x, rdotr)
#
#     p, r, x, rdotr = tf.while_loop(body=body, cond=cond, loop_vars=loop_vars, maximum_iterations=cg_iters)
#     accept = tf.less(rdotr, residual_tol)
#     # accept = tf.Print(accept, ['accepted', accept])
#     return accept, x
