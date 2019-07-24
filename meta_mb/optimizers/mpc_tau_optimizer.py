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
        # self._global_norms = []
        self.y_for_plot = []

    def build_graph(self, loss, var_list, result_op, input_ph_dict, *args, **kwargs):
        assert isinstance(loss, tf.Tensor)
        assert isinstance(input_ph_dict, dict)

        self._input_ph_dict = input_ph_dict
        self._loss = loss
        self.with_policy = kwargs.get('with_policy', False)

        grads_vars = self._tf_optimizer.compute_gradients(loss, var_list=var_list)
        grads, vars = zip(*grads_vars)

        if self.with_policy:
            grads, self._grads_for_plot = tf.clip_by_global_norm(grads, clip_norm=5)
        else:
            assert 'mean' in vars[0].name
            grad_mean = grads[0]
            # FIXME: won't work with parametrized policy
            # grads[0] = grad_mean has shape (horizon, num_rollouts, act_space_dims)_
            # take the norm of the gradient of the first rollout
            self._grads_for_plot = tf.norm(grad_mean[:, 0, :], axis=-1)  # (horizon,)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=5)

        self._train_op = self._tf_optimizer.apply_gradients(zip(grads, vars))

        self.save_dir = os.path.join(logger.get_dir(), 'grads_global_norm')
        os.makedirs(self.save_dir, exist_ok=True)

        # Dual gradient descent
        # if 'lmbda' in kwargs:
        #     with tf.control_dependencies([self._train_op, tf.print(kwargs['lmbda'], self._loss + kwargs['loss_dual'], kwargs['loss_dual'])]):
        #         self._train_op_dual = tf.no_op()
        #     #     self._train_op_dual = self._tf_optimizer.minimize(kwargs['loss_dual'], var_list=[kwargs['lmbda']])
        # else:
        #     self._train_op_dual = tf.no_op()

        if 'init_op' in kwargs:
            self._init_op = kwargs['init_op']
        if 'extra_result_op' in kwargs:
            self._extra_result_op = kwargs['extra_result_op']

        self._result_op = result_op

    def loss(self, input_val_dict):
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        loss = sess.run(self._loss, feed_dict=feed_dict)
        return loss

    def optimize(self, input_val_dict, run_extra_result_op=False, log_grads_for_plot=False):
        self._global_step += 1
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        if self._init_op is not None:
            sess.run(self._init_op, feed_dict=feed_dict)

        array_grads_for_plot = []
        loss_array = []
        for epoch in range(self._max_epochs):
            loss, _, grads_for_plot = sess.run([self._loss, self._train_op, self._grads_for_plot], feed_dict=feed_dict)
            loss_array.append(loss)
            array_grads_for_plot.append(grads_for_plot)
        #  array_grads_for_plot = (opt_epochs, horizon)
        # grads_for_plot = np.mean(array_grads_for_plot, axis=0)  # mean with respect to optimization epochs

        # if log_global_norms:
        #     self._global_norms.append(array_grads_for_plot)  # global_norms = (max_epochs,)
        #     # logger.log(loss_array)
        if log_grads_for_plot:
            self.y_for_plot.append(array_grads_for_plot)

        if run_extra_result_op:
            result = sess.run(self._result_op + self._extra_result_op, feed_dict)
        else:
            result = sess.run(self._result_op, feed_dict)

        return result

    def plot_grads(self):
        y_for_plot = np.stack(self.y_for_plot, axis=-1)  # (opt_epochs, horizon, max_path_length)
        self.y_for_plot = []
        logger.log('plt array has size', y_for_plot.shape)

        # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(50, 30))

        # plot path_length (x-axis) vs. opt_epochs (y-axis)
        # y = np.mean(y_for_plot, axis=1)
        # im = axes[0].imshow(y, cmap='hot', interpolation='nearest')
        # axes[0].set_xlabel('path_length_collected_by sampler_so_far')
        # axes[0].set_ylabel('opt_epochs')
        # fig.colorbar(im, ax=axes[0])

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(50, 30))
        # plot path_length (x-axis) vs. horizon (y-axis)
        y = np.mean(y_for_plot, axis=0)
        im = ax.imshow(y, cmap='hot', interpolation='nearest')
        ax.set_xlabel('path_length_collected_by_sampler_so_far')
        ax.set_ylabel('horizon')
        fig.colorbar(im, ax=ax)

        # plt.show()
        fig.suptitle(f'{self._global_step}')
        fig.savefig(os.path.join(self.save_dir, f'{self._global_step}.png'))
        logger.log('plt saved to', os.path.join(self.save_dir, f'{self._global_step}.png'))

