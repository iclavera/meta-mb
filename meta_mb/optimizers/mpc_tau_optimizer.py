from meta_mb.logger import logger
import matplotlib.pyplot as plt
import numpy as np
from meta_mb.optimizers.base import Optimizer
from meta_mb.utils import Serializable
import tensorflow as tf
import os


class MPCTauOptimizer(Optimizer, Serializable):
    def __init__(
            self,
            clip_norm,
            learning_rate,
            max_epochs,
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=None,
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
        self.y1_for_plot = []
        self.y2_for_plot = []
        self.clip_norm = clip_norm

    def build_graph(self, loss, var_list, result_op, input_ph_dict, *args, **kwargs):
        assert isinstance(loss, tf.Tensor)
        assert isinstance(input_ph_dict, dict)

        self._input_ph_dict = input_ph_dict
        self._loss = loss
        self.with_policy = kwargs.get('with_policy', False)

        grads_vars = self._tf_optimizer.compute_gradients(loss, var_list=var_list)
        grads, vars = zip(*grads_vars)

        if self.clip_norm > 0:
            grads, grads_global_norm = tf.clip_by_global_norm(grads, clip_norm=self.clip_norm)
        else:
            grads_global_norm = tf.linalg.global_norm(grads)

        if self.with_policy:
            self._grads_for_plot = grads_global_norm  # /(len(vars) * 0.5)  # shape = ()
        else:
            assert 'mean' in vars[0].name
            grad_mean = grads[0]
            self._grads_for_plot = tf.norm(grad_mean[:, 0, :], axis=-1)  # (horizon,)  after clipping
        self._vars_for_plot = tf.linalg.global_norm(var_list)
        self._len_var_list = len(var_list)

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
        array_vars_for_plot = []
        loss_array = []
        for epoch in range(self._max_epochs):
            loss, _, grads_for_plot, vars_for_plot = sess.run([self._loss, self._train_op, self._grads_for_plot, self._vars_for_plot],
                                                              feed_dict=feed_dict)
            loss_array.append(loss)
            array_grads_for_plot.append(grads_for_plot)
            array_vars_for_plot.append(vars_for_plot)

        if log_grads_for_plot:
            self.y1_for_plot.append(array_grads_for_plot)
            self.y2_for_plot.append(array_vars_for_plot)

        if run_extra_result_op:
            result = sess.run(self._result_op + self._extra_result_op, feed_dict)
        else:
            result = sess.run(self._result_op, feed_dict)

        return result

    def plot_grads(self):
        y1 = np.stack(self.y1_for_plot, axis=-1)  # (opt_epochs, [horizon, ]max_path_length)
        y2 = np.stack(self.y2_for_plot, axis=-1)  # (opt_epochs, max_path_length)
        self.y1_for_plot, self.y2_for_plot = [], []
        logger.log('plt array has size', y1.shape, y2.shape)
        logger.log('num of vars = ', self._len_var_list)

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30, 30))

        # plot path_length (x-axis) vs. horizon (y-axis)
        ax = axes[0]
        if not self.with_policy:
            y1 = np.mean(y1, axis=0)
        im = ax.imshow(y1, cmap='hot', interpolation='nearest')
        ax.set_xlabel('path_length_collected_by_sampler_so_far')
        ax.set_ylabel('horizon')
        ax.set_title('global_norm(grads)')
        fig.colorbar(im, ax=ax)

        # plot path_length (x-axis) vs. opt_epochs (y-axis)
        ax = axes[1]
        im = ax.imshow(y2, cmap='hot', interpolation='nearest')
        ax.set_xlabel('path_length_collected_by sampler_so_far')
        ax.set_ylabel('opt_epochs')
        ax.set_title('global_norm(vars)')
        fig.colorbar(im, ax=ax)

        # plt.show()
        fig.suptitle(f'{self._global_step}')
        fig.savefig(os.path.join(self.save_dir, f'{self._global_step}.png'))
        logger.log('plt saved to', os.path.join(self.save_dir, f'{self._global_step}.png'))

