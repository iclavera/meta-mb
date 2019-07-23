from meta_mb.logger import logger
import matplotlib.pyplot as plt
import numpy as np
from meta_mb.optimizers.base import Optimizer
from meta_mb.utils import Serializable
import tensorflow as tf
from time import time
import os


class MPCTauOptimizer(Optimizer, Serializable):
    def __init__(
            self,
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=None,
            learning_rate=1e-3,
            max_epochs=1,
            verbose=False,
            do_plots=False,
    ):
        Serializable.quick_init(self, locals())
        if tf_optimizer_args is None:
            tf_optimizer_args = dict()
        tf_optimizer_args['learning_rate'] = learning_rate

        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._verbose = verbose
        self._train_op = None
        self._train_op_dual = None
        self._loss = None
        self._input_ph_dict = None
        self._init_op = None
        self._result_op = None
        self._global_step = 0
        self.do_plots = do_plots

    def build_graph(self, loss, var_list, result_op, input_ph_dict, *args, **kwargs):
        assert isinstance(loss, tf.Tensor)
        assert isinstance(input_ph_dict, dict)

        self._input_ph_dict = input_ph_dict
        self._loss = loss

        grads_vars = self._tf_optimizer.compute_gradients(loss, var_list=var_list)
        grads, vars = zip(*grads_vars)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=5)
        self._train_op = self._tf_optimizer.apply_gradients(zip(grads, vars))
        # Plotting: gradient of tau_mean for the first obs_sample in first action space dimension
        if self.do_plots:
            self.grad_sample = tf.transpose(grads[0], perm=[1, 2, 0])[0][0]  # (horizon,)
            self.save_dir = os.path.join(logger.get_dir(), 'grads_first_act_dim')
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.grad_sample = tf.no_op()

        # if 'lmbda' in kwargs:
        #     with tf.control_dependencies([self._train_op, tf.print(kwargs['lmbda'], self._loss + kwargs['loss_dual'], kwargs['loss_dual'])]):
        #         self._train_op_dual = tf.no_op()
        #     #     self._train_op_dual = self._tf_optimizer.minimize(kwargs['loss_dual'], var_list=[kwargs['lmbda']])
        # else:
        #     self._train_op_dual = tf.no_op()

        if 'init_op' in kwargs:
            self._init_op = kwargs['init_op']

        self._result_op = result_op

    def loss(self, input_val_dict):
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        loss = sess.run(self._loss, feed_dict=feed_dict)
        return loss

    def optimize(self, input_val_dict, do_plots=False):
        self._global_step += 1
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        if self._init_op is not None:
            sess.run(self._init_op, feed_dict=feed_dict)

        loss_before_opt = None
        loss_array = []
        grads_mean_tt = []
        for epoch in range(self._max_epochs):
            loss, _, grad_sample = sess.run([self._loss, self._train_op, self.grad_sample], feed_dict=feed_dict)
            grads_mean_tt.append(grad_sample)

            #if not loss_before_opt: loss_before_opt = loss
            loss_array.append(loss)

        if self._verbose:
            loss_array = np.stack(loss_array, axis=-1)
            print(loss_array[0])

        # plotting
         # (max_epochs, horizon)
        if self.do_plots and do_plots:
            fig, ax = plt.subplots()
            im = ax.imshow(grads_mean_tt, cmap='hot', interpolation='nearest')
            ax.set_xticklabels(np.arange(len(grads_mean_tt[0])))
            ax.set_yticklabels(np.arange(len(grads_mean_tt)))
            ax.set_title(f'{self._global_step}')
            fig.colorbar(im, ax=ax)
            fig.savefig(os.path.join(self.save_dir, f'{self._global_step}.png'))
            logger.log('plt saved to', os.path.join(self.save_dir, f'{self._global_step}.png'))

        result = sess.run(self._result_op, feed_dict)

        return result

