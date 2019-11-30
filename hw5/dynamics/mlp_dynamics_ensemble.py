from hw5.dynamics.layers import MLP
import tensorflow as tf
import numpy as np
from hw5.utils.serializable import Serializable
from hw5.utils import compile_function
from hw5.logger import logger
from hw5.dynamics.mlp_dynamics import MLPDynamicsModel
import time
from collections import OrderedDict
from hw5.dynamics.utils import normalize, denormalize, train_test_split


class MLPDynamicsEnsemble(MLPDynamicsModel):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 num_models=5,
                 hidden_sizes=(512, 512),
                 hidden_nonlinearity='swish',
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 weight_normalization=False,  # Doesn't work
                 normalize_input=True,
                 optimizer=tf.train.AdamOptimizer,
                 valid_split_ratio=0.2,  # 0.1
                 rolling_average_persitency=0.99,
                 buffer_size=50000,
                 loss_str='MSE',
                 ):

        Serializable.quick_init(self, locals())

        max_logvar = 1
        min_logvar = 0.1

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.buffer_size_train = int(buffer_size * (1 - valid_split_ratio))
        self.buffer_size_test = int(buffer_size * valid_split_ratio)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_models = num_models
        self.hidden_sizes = hidden_sizes
        self.name = name
        self._dataset_train = None
        self._dataset_test = None

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]
        self.timesteps_counter = 0
        self.used_timesteps_counter = 0

        self.hidden_nonlinearity = hidden_nonlinearity = self._activations[hidden_nonlinearity]
        self.output_nonlinearity = output_nonlinearity = self._activations[output_nonlinearity]

        """ computation graph for training and simple inference """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

            self._create_stats_vars()

            # concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)

            obs_ph = tf.split(self.nn_input, self.num_models, axis=0)

            # create MLP
            mlps = []
            delta_preds = []
            self.obs_next_pred = []
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=tf.AUTO_REUSE):
                    mlp = MLP(name+'/model_{}'.format(i),
                              output_dim=obs_space_dims,
                              hidden_sizes=hidden_sizes,
                              hidden_nonlinearity=hidden_nonlinearity,
                              output_nonlinearity=output_nonlinearity,
                              input_var=obs_ph[i],
                              input_dim=obs_space_dims+action_space_dims,
                              )
                    mlps.append(mlp)

                delta_preds.append(mlp.output_var)

            self.delta_pred = tf.stack(delta_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)


            # define loss and train_op
            if loss_str == 'L2':
                self.loss = tf.reduce_mean(tf.linalg.norm(self.delta_ph[:, :, None] - self.delta_pred, axis=1))
            elif loss_str == 'MSE':
                self.loss = tf.reduce_mean((self.delta_ph[:, :, None] - self.delta_pred)**2)
            else:
                raise NotImplementedError

            self.optimizer = optimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            # tensor_utils
            self.f_delta_pred = compile_function([self.obs_ph, self.act_ph], self.delta_pred)

        """ computation graph for inference where each of the models receives a different batch"""
        with tf.variable_scope(name, reuse=True):
            # placeholders
            self.obs_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

            # split stack into the batches for each model --> assume each model receives a batch of the same size
            self.obs_model_batches = tf.split(self.obs_model_batches_stack_ph, self.num_models, axis=0)
            self.act_model_batches = tf.split(self.act_model_batches_stack_ph, self.num_models, axis=0)
            self.delta_model_batches = tf.split(self.delta_model_batches_stack_ph, self.num_models, axis=0)

            # reuse previously created MLP but each model receives its own batch
            delta_preds = []
            self.obs_next_pred = []
            self.loss_model_batches = []
            self.train_op_model_batches = []
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):
                    # concatenate action and observation --> NN input
                    nn_input = tf.concat([self.obs_model_batches[i], self.act_model_batches[i]], axis=1)
                    mlp = MLP(name+'/model_{}'.format(i),
                              output_dim=obs_space_dims,
                              hidden_sizes=hidden_sizes,
                              hidden_nonlinearity=hidden_nonlinearity,
                              output_nonlinearity=output_nonlinearity,
                              input_var=nn_input,
                              input_dim=obs_space_dims+action_space_dims,
                              weight_normalization=weight_normalization)

                delta_preds.append(mlp.output_var)
                if loss_str == 'L2':
                    loss = tf.reduce_mean(tf.linalg.norm(self.delta_model_batches[i] - mlp.output_var, axis=1))
                elif loss_str == 'MSE':
                    loss = tf.reduce_mean((self.delta_model_batches[i] - mlp.output_var) ** 2)
                else:
                    raise  NotImplementedError
                self.loss_model_batches.append(loss)
                self.train_op_model_batches.append(optimizer(learning_rate=self.learning_rate).minimize(loss))
            self.delta_pred_model_batches_stack = tf.concat(delta_preds, axis=0) # shape: (batch_size_per_model*num_models, ndim_obs)

            # tensor_utils
            self.f_delta_pred_model_batches = compile_function([self.obs_model_batches_stack_ph,
                                                                self.act_model_batches_stack_ph],
                                                                self.delta_pred_model_batches_stack)

        self._networks = mlps
        # LayersPowered.__init__(self, [mlp.output_layer for mlp in mlps])

    """YOUR CODE HERE FOR PROBLEM 2C"""
    # hint: you should find the single model and look for what functions you need to implement.





