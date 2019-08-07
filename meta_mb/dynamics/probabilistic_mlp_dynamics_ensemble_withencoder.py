from meta_mb.dynamics.layers import MLP
import tensorflow as tf
import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.dynamics.mlp_dynamics_ensemble_withencoder import MLPDynamicsEnsemble


class ProbMLPDynamicsEnsemble(MLPDynamicsEnsemble):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 num_stack,
                 encoder,
                 input_is_img,
                 latent_dim,
                 buffer,
                 model_grad_thru_enc,
                 num_models=5,
                 hidden_sizes=(512, 512),
                 hidden_nonlinearity='swish',
                 output_nonlinearity=None,
                 batch_size=500,
                 learning_rate=0.001,
                 weight_normalization=False,  # Doesn't work
                 normalize_input=True,
                 optimizer=tf.train.AdamOptimizer,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 buffer_size=50000,
                 cpc_loss_weight=0.,
                 cpc_model=None,
                 ):

        Serializable.quick_init(self, locals())

        max_logvar = .5
        min_logvar = -10

        self.num_stack = num_stack
        self.encoder = encoder
        self.input_is_img = input_is_img
        self.cpc_model = cpc_model
        self.cpc_loss_weight = cpc_loss_weight
        self.latent_dim = latent_dim
        self.buffer = buffer
        self.model_grad_thru_enc = model_grad_thru_enc

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
        self.name = name
        self.hidden_sizes = hidden_sizes

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape
        self.action_space_dims = action_space_dims = env.action_space.shape[0]
        self.timesteps_counter = 0
        self.used_timesteps_counter = 0

        hidden_nonlinearity = self._activations[hidden_nonlinearity]
        output_nonlinearity = self._activations[output_nonlinearity]
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        """ computation graph for training and simple inference """
        # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        #     self._create_stats_vars()
        #
        #     self.max_logvar = tf.Variable(np.ones([1, obs_space_dims]) * max_logvar, dtype=tf.float32,
        #                                   trainable=True,
        #                                   name="max_logvar")
        #     self.min_logvar = tf.Variable(np.ones([1, obs_space_dims]) * min_logvar, dtype=tf.float32,
        #                                   trainable=True,
        #                                   name="min_logvar")
        #     self._create_assign_ph()
        #
        #     # placeholders
        #     self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
        #     self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
        #     self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
        #
        #     # concatenate action and observation --> NN input
        #     self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)
        #
        #     obs_ph = tf.split(self.nn_input, self.num_models, axis=0)
        #
        #     # create MLP
        #     mlps = []
        #     delta_preds = []
        #     var_preds = []
        #     logvar_preds = []
        #     invar_preds = []
        #     self.obs_next_pred = []
        #     for i in range(num_models):
        #         with tf.variable_scope('model_{}'.format(i)):
        #             mlp = MLP(name+'/model_{}'.format(i),
        #                       output_dim=2 * obs_space_dims,
        #                       hidden_sizes=hidden_sizes,
        #                       hidden_nonlinearity=hidden_nonlinearity,
        #                       output_nonlinearity=output_nonlinearity,
        #                       input_var=obs_ph[i],
        #                       input_dim=obs_space_dims+action_space_dims, # FIXME: input weight_normalization?
        #                       )
        #             mlps.append(mlp)
        #
        #         mean, logvar = tf.split(mlp.output_var, 2,  axis=-1)
        #         logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
        #         logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        #         var = tf.exp(logvar)
        #         inv_var = tf.exp(-logvar)
        #
        #         delta_preds.append(mean)
        #         logvar_preds.append(logvar)
        #         var_preds.append(var)
        #         invar_preds.append(inv_var)
        #
        #     self.delta_pred = tf.stack(delta_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)
        #     self.logvar_pred = tf.stack(logvar_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)
        #     self.var_pred = tf.stack(var_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)
        #     self.invar_pred = tf.stack(invar_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)
        #
        #     # define loss and train_op
        #     self.loss = tf.reduce_mean(tf.square(self.delta_ph[:, :, None] - self.delta_pred) * self.invar_pred
        #                                + self.logvar_pred)
        #     self.loss += 0.01 * tf.reduce_mean(self.max_logvar) - 0.01 * tf.reduce_mean(self.min_logvar)
        #     self.optimizer = optimizer(learning_rate=self.learning_rate)
        #     self.train_op = self.optimizer.minimize(self.loss)
        #
        #     # tensor_utils
        #     self.f_delta_pred = compile_function([self.obs_ph, self.act_ph], self.delta_pred)
        #     self.f_var_pred = compile_function([self.obs_ph, self.act_ph], self.var_pred)

        """ computation graph for inference where each of the models receives a different batch"""
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.max_logvar = tf.Variable(np.ones([1, latent_dim]) * max_logvar, dtype=tf.float32,
                                          trainable=True,
                                          name="max_logvar")
            self.min_logvar = tf.Variable(np.ones([1, latent_dim]) * min_logvar, dtype=tf.float32,
                                          trainable=True,
                                          name="min_logvar")
            self._create_assign_ph()

            if not input_is_img:
                self.obs_ph = obs = tf.placeholder(tf.float32, shape=(None, num_stack, latent_dim))
                self.act_ph = act = tf.placeholder(tf.float32, shape=(None, action_space_dims))
                self.delta_ph = delta = tf.placeholder(tf.float32, shape=(None, latent_dim))

            else:
                self.obs_ph = obs = tf.placeholder(tf.float32, shape=(None, num_stack, *obs_space_dims))
                self.act_ph = act = tf.placeholder(tf.float32, shape=(None, action_space_dims))
                self.delta_ph = delta = tf.placeholder(tf.float32, shape=(None, *obs_space_dims))
                if self.model_grad_thru_enc:
                    obs = self.encoder(tf.reshape(obs, shape=(-1, *obs_space_dims)))
                    obs = tf.reshape(obs, shape=(-1, num_stack * latent_dim))
                    delta = self.encoder(delta) - obs[:, -latent_dim:]
                else:
                    obs = tf.stop_gradient(self.encoder(tf.reshape(obs, shape=(-1, *obs_space_dims))))
                    obs = tf.reshape(obs, shape=(-1, num_stack * latent_dim))
                    delta = tf.stop_gradient(self.encoder(delta) - obs[:, -latent_dim:])

            # split stack into the batches for each model --> assume each model receives a batch of the same size
            self.obs_model_batches = tf.split(obs, self.num_models, axis=0)
            self.act_model_batches = tf.split(act, self.num_models, axis=0)
            self.delta_model_batches = tf.split(delta, self.num_models, axis=0)

            # reuse previously created MLP but each model receives its own batch
            mlps = []
            delta_preds = []
            var_preds = []
            self.obs_next_pred = []
            self.model_loss = []

            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=tf.AUTO_REUSE):
                    # concatenate action and observation --> NN input
                    nn_input = tf.concat([tf.reshape(self.obs_model_batches[i], shape=(-1, num_stack * latent_dim)),
                                          self.act_model_batches[i]], axis=1)
                    mlp = MLP(name+'/model_{}'.format(i),
                              output_dim=2 * latent_dim,
                              hidden_sizes=hidden_sizes,
                              hidden_nonlinearity=hidden_nonlinearity,
                              output_nonlinearity=output_nonlinearity,
                              input_var=nn_input,
                              input_dim=latent_dim * num_stack +action_space_dims,
                              weight_normalization=weight_normalization)
                    mlps.append(mlp)

                mean, logvar = tf.split(mlp.output_var, 2, axis=-1)
                logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
                logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
                var = tf.exp(logvar)
                inv_var = tf.exp(-logvar)

                loss = tf.reduce_mean(tf.square(self.delta_model_batches[i] - mean) * inv_var + logvar)
                loss += (0.01 * tf.reduce_mean(self.max_logvar) - 0.01 * tf.reduce_mean(self.min_logvar))

                delta_preds.append(mean)
                var_preds.append(var)
                self.model_loss.append(loss)

            self.total_loss = 1 / self.num_models * sum(self.model_loss)
            if cpc_loss_weight > 0:
                self.total_loss += cpc_loss_weight * self.cpc_model.loss
            self.optimizer = optimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.total_loss)


        self._networks = mlps

    """

    def predict_sym(self, obs_ph, act_ph):
        delta_preds = []
        logvar_preds = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for i in range(self.num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):
                    assert self.normalize_input
                    in_obs_var = (obs_ph - self._mean_obs_var[i])/(self._std_obs_var[i] + 1e-8)
                    in_act_var = (act_ph - self._mean_act_var[i]) / (self._std_act_var[i] + 1e-8)
                    input_var = tf.concat([in_obs_var, in_act_var], axis=1)
                    mlp = MLP(self.name+'/model_{}'.format(i),
                              output_dim=2 * self.obs_space_dims,
                              hidden_sizes=self.hidden_sizes,
                              hidden_nonlinearity=self.hidden_nonlinearity,
                              output_nonlinearity=self.output_nonlinearity,
                              input_var=input_var,
                              input_dim=self.obs_space_dims + self.action_space_dims,
                              )

                mean, logvar = tf.split(mlp.output_var, 2, axis=-1)
                logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
                logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)

                delta_preds.append(mean)
                logvar_preds.append(logvar)

        delta_pred = tf.stack(delta_preds, axis=1)  # shape: (batch_size, n_models, ndim_obs)
        logvar_pred = tf.stack(logvar_preds, axis=1)  # shape: (batch_size, n_models, ndim_obs)
        delta_pred = delta_pred + tf.random.normal(shape=tf.shape(delta_pred)) * tf.exp(logvar_pred * 0.5)
        delta_pred = tf.clip_by_value(delta_pred, -1e3, 1e3)

        # select one model for each row
        model_idx = tf.random.uniform(shape=(tf.shape(delta_pred)[0],), minval=0, maxval=self.num_models, dtype=tf.int32)
        delta_pred = tf.batch_gather(delta_pred, tf.reshape(model_idx, [-1, 1]))
        delta_pred = tf.squeeze(delta_pred, axis=1)

        return obs_ph + delta_pred
        
    def distribution_info_sym(self, obs_var, act_var):
        means = []
        log_stds = []
        with tf.variable_scope(self.name, reuse=True):
            obs_var = tf.split(obs_var, self.num_models, axis=0)
            act_var = tf.split(act_var, self.num_models, axis=0)
            for i in range(self.num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):
                    if self.normalize_input:
                        in_obs_var = (obs_var[i] - self._mean_obs_var[i])/(self._std_obs_var[i] + 1e-8)
                        in_act_var = (act_var[i] - self._mean_act_var[i]) / (self._std_act_var[i] + 1e-8)
                    else:
                        in_obs_var = obs_var[i]
                        in_act_var = act_var[i]
                    input_var = tf.concat([in_obs_var, in_act_var], axis=1)
                    mlp = MLP(self.name+'/model_{}'.format(i),
                              output_dim=2 * self.obs_space_dims,
                              hidden_sizes=self.hidden_sizes,
                              hidden_nonlinearity=self.hidden_nonlinearity,
                              output_nonlinearity=self.output_nonlinearity,
                              input_var=input_var,
                              input_dim=self.obs_space_dims + self.action_space_dims,
                              )

                mean, logvar = tf.split(mlp.output_var, 2, axis=-1)
                logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
                logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)

                means.append(mean)
                log_stds.append(logvar/2)

        mean = tf.concat(means, axis=0)
        mean = tf.clip_by_value(mean, -1e3, 1e3)  #?
        log_std = tf.concat(log_stds, axis=0)
        return dict(mean=mean, var=log_std)
    """

    def predict_sym(self, obs_ph, act_ph):
        """
        Same batch fed into all models. Randomly output one of the predictions for each observation.
        :param obs_ph: (batch_size, obs_space_dims)
        :param act_ph: (batch_size, act_space_dims)
        :return: (batch_size, obs_space_dims)
        """
        original_obs = obs_ph
        if self.normalize_input:
            obs_ph = (obs_ph - self.buffer._mean_obs_var)/(self.buffer._std_obs_var + 1e-8)
            act_ph = (act_ph - self.buffer._mean_act_var)/(self.buffer._std_act_var + 1e-8)

        obs_ph = tf.reshape(obs_ph, shape=(-1, self.num_stack * self.latent_dim))

        # shuffle
        perm = tf.range(0, limit=tf.shape(obs_ph)[0], dtype=tf.int32)
        perm = tf.random.shuffle(perm)
        obs_ph, act_ph = tf.gather(obs_ph, perm), tf.gather(act_ph, perm)
        obs_ph, act_ph = tf.split(obs_ph, self.num_models, axis=0), tf.split(act_ph, self.num_models, axis=0)

        delta_preds = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for i in range(self.num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):

                    input_var = tf.concat([obs_ph[i], act_ph[i]], axis=1)
                    mlp = MLP(self.name+'/model_{}'.format(i),
                              output_dim=2 * self.latent_dim,
                              hidden_sizes=self.hidden_sizes,
                              hidden_nonlinearity=self.hidden_nonlinearity,
                              output_nonlinearity=self.output_nonlinearity,
                              input_var=input_var,
                              input_dim=self.latent_dim * self.num_stack + self.action_space_dims,
                              )

                    mean, logvar = tf.split(mlp.output_var, 2, axis=-1)
                    logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
                    logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
                    delta_pred = mean + tf.random.normal(shape=tf.shape(mean)) * tf.exp(logvar)
                    # denormalize
                    if self.normalize_input:
                        delta_pred = delta_pred * self.buffer._std_delta_var[i] + self.buffer._mean_delta_var[i]
                    delta_preds.append(delta_pred)

        delta_preds = tf.concat(delta_preds, axis=0)

        # unshuffle
        perm_inv = tf.invert_permutation(perm)
        next_obs = original_obs[:, -1, :] + tf.gather(delta_preds, perm_inv)
        next_obs = tf.clip_by_value(next_obs, -1e2, 1e2)
        return tf.concat([original_obs[:, 1:], next_obs[:, None, :]], axis=1)



    def _create_assign_ph(self):
        self._min_log_var_ph = tf.placeholder(tf.float32, shape=[1, self.latent_dim], name="min_logvar_ph")
        self._max_log_var_ph = tf.placeholder(tf.float32, shape=[1, self.latent_dim], name="max_logvar_ph")
        self._assign_ops_var = [tf.assign(self.min_logvar, self._min_log_var_ph), tf.assign(self.max_logvar,
                                                                                            self._max_log_var_ph)]

    def __getstate__(self):
        state = MLPDynamicsEnsemble.__getstate__(self)
        sess = tf.get_default_session()
        state['min_log_var'] = sess.run(self.min_logvar)
        state['max_log_var'] = sess.run(self.max_logvar)
        return state

    def __setstate__(self, state):
        MLPDynamicsEnsemble.__setstate__(self, state)
        sess = tf.get_default_session()
        sess.run(self._assign_ops_var, feed_dict={self._min_log_var_ph: state['min_log_var'],
                                                  self._max_log_var_ph: state['max_log_var']
                                                  })

    def get_shared_param_values(self): # to feed policy
        state = MLPDynamicsEnsemble.get_shared_param_values(self)
        sess = tf.get_default_session()
        state['min_log_var'] = sess.run(self.min_logvar)
        state['max_log_var'] = sess.run(self.max_logvar)
        return state

    def set_shared_params(self, state):
        MLPDynamicsEnsemble.set_shared_params(self, state)
        sess = tf.get_default_session()
        sess.run(self._assign_ops_var, feed_dict={
            self._min_log_var_ph: state['min_log_var'],
            self._max_log_var_ph: state['max_log_var'],
        })

def denormalize(data_array, mean, std):
    if data_array.ndim == 3: # assumed shape (batch_size, ndim_obs, n_models)
        return data_array * (std[None, :, None] + 1e-10) + mean[None, :, None]
    elif data_array.ndim == 2:
        return data_array * (std[None, :] + 1e-10) + mean[None, :]
