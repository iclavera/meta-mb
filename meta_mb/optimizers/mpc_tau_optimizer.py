from meta_mb.logger import logger
import numpy as np
from meta_mb.optimizers.base import Optimizer
from meta_mb.utils import Serializable
import tensorflow as tf


class MPCTauOptimizer(Optimizer, Serializable):
    def __init__(
            self,
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=None,
            learning_rate=1e-3,
            max_epochs=1,
            verbose=False,
    ):
        Serializable.quick_init(self, locals())
        if tf_optimizer_args is None:
            tf_optimizer_args = dict()
        tf_optimizer_args['learning_rate'] = learning_rate

        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._verbose = verbose
        self._train_op = None
        self._loss = None
        self._input_ph_dict = None

    def build_graph(self, loss, var_list, init_op, result_op, input_ph_dict, *args, **kwargs):
        assert isinstance(loss, tf.Tensor)
        assert isinstance(input_ph_dict, dict)

        self._input_ph_dict = input_ph_dict
        self._loss = loss
        self._train_op = self._tf_optimizer.minimize(loss, var_list=var_list)
        self._init_op = init_op
        self._result_op = result_op

    def loss(self, input_val_dict):
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        loss = sess.run(self._loss, feed_dict=feed_dict)
        return loss

    def optimize(self, input_val_dict):
        sess = tf.get_default_session()
        sess.run(self._init_op)
        feed_dict = self.create_feed_dict(input_val_dict)

        loss_before_opt = None
        loss_array = []
        for epoch in range(self._max_epochs):
            loss, _ = sess.run([self._loss, self._train_op], feed_dict)

            #if not loss_before_opt: loss_before_opt = loss

            loss_array.append(loss)

        if self._verbose:
            loss_array = np.stack(loss_array, axis=-1)
            print(loss_array[:, 1:] - loss_array[:, :-1])

        result = sess.run(self._result_op, feed_dict)

        return result

