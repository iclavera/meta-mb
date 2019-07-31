from meta_mb.dynamics.layers import MLP
import tensorflow as tf
import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.dynamics.mlp_dynamics_q import MLPDynamicsEnsembleQ
from meta_mb.utils.utils import remove_scope_from_name
from pdb import set_trace as st
from collections import OrderedDict

class ProbMLPDynamicsEnsembleQ(MLPDynamicsEnsembleQ):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 Qs,
                 policy,
                 reward_scale=1,
                 discount = 1,
                 Q_targets=None,
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
                 q_loss_importance=0,
                 ):

        Serializable.quick_init(self, locals())

        max_logvar = .5
        min_logvar = -10

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
        self._dataset_train = None
        self._dataset_test = None
        self.env = env
        self.Qs = Qs
        if Q_targets is None:
            self.Q_targets = Qs
        else:
            self.Q_targets = Q_targets
        self.policy = policy
        self.reward_scale = reward_scale
        self.discount = discount

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]
        self.timesteps_counter = 0
        self.used_timesteps_counter = 0
        self.reward_scale = reward_scale
        self.discount=discount
        self.q_loss_importance = q_loss_importance

        hidden_nonlinearity = self._activations[hidden_nonlinearity]
        output_nonlinearity = self._activations[output_nonlinearity]
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        """ computation graph for training and simple inference """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._create_stats_vars()

            self.max_logvar = tf.Variable(np.ones([1, obs_space_dims]) * max_logvar, dtype=tf.float32,
                                          trainable=True,
                                          name="max_logvar")
            self.min_logvar = tf.Variable(np.ones([1, obs_space_dims]) * min_logvar, dtype=tf.float32,
                                          trainable=True,
                                          name="min_logvar")
            self._create_assign_ph()
            log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
            alpha = tf.exp(log_alpha)
            self.log_alpha = log_alpha

            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.rewards_ph = tf.placeholder(tf.float32, shape=(None))
            self.dones_ph = tf.placeholder(tf.bool, shape=(None))

            # concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)

            obs_ph = tf.split(self.nn_input, self.num_models, axis=0)

            # create MLP
            mlps = []
            delta_preds = []
            var_preds = []
            logvar_preds = []
            invar_preds = []
            self.obs_next_pred = []
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i)):
                    mlp = MLP(name+'/model_{}'.format(i),
                              output_dim=2 * obs_space_dims,
                              hidden_sizes=hidden_sizes,
                              hidden_nonlinearity=hidden_nonlinearity,
                              output_nonlinearity=output_nonlinearity,
                              input_var=obs_ph[i],
                              input_dim=obs_space_dims+action_space_dims, # FIXME: input weight_normalization?
                              )
                    mlps.append(mlp)

                mean, logvar = tf.split(mlp.output_var, 2,  axis=-1)
                logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
                logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
                var = tf.exp(logvar)
                inv_var = tf.exp(-logvar)

                delta_preds.append(mean)
                logvar_preds.append(logvar)
                var_preds.append(var)
                invar_preds.append(inv_var)

            self.delta_pred = tf.stack(delta_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)
            self.logvar_pred = tf.stack(logvar_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)
            self.var_pred = tf.stack(var_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)
            self.invar_pred = tf.stack(invar_preds, axis=2)  # shape: (batch_size, ndim_obs, n_models)

            # input_q_fun = tf.concat([self.obs_ph, self.act_ph], axis=-1)

            next_obs = self.predict_sym(self.obs_ph, self.act_ph)
            dist_info_sym = self.policy.distribution_info_sym(next_obs)
            next_actions_var, dist_info_sym = self.policy.distribution.sample_sym(dist_info_sym)
            # next_log_pis_var = self.policy.distribution.log_likelihood_sym(next_actions_var, dist_info_sym)
            # next_log_pis_var = tf.expand_dims(next_log_pis_var, axis=-1)
            input_q_fun = tf.concat([next_obs, next_actions_var], axis=-1)
            next_q_values = [Q.value_sym(input_var=input_q_fun) for Q in self.Qs]

            min_next_Q = tf.reduce_min(next_q_values, axis=0)
            # change this for other environments
            dones_ph = tf.cast(self.dones_ph, self.obs_ph.dtype)
            dones_ph = tf.expand_dims(dones_ph, axis=-1)
            # rewards_ph = tf.expand_dims(self.rewards_ph, axis=-1)
            rewards = self.env.tf_reward(self.obs_ph, self.act_ph, next_obs)
            input_q_fun = tf.concat([next_obs, next_actions_var], axis=-1)
            q_values = [rewards + self.discount * (1 - dones_ph) * Q.value_sym(input_var=input_q_fun) for Q in self.Qs]

            # define loss and train_op
            input_q_fun = tf.concat([self.obs_ph, self.act_ph], axis=-1)

            q_values_var = [Q.value_sym(input_var=input_q_fun) for Q in self.Qs]
            q_losses = [tf.losses.mean_squared_error(labels=q_values[i], predictions=q_values_var[i], weights=0.5)
                                        for i in range(2)]

            self.loss = [tf.reduce_mean(tf.square(self.delta_ph[:, :, None] - self.delta_pred) * self.invar_pred
                                       + self.logvar_pred)
                                       + 0.01 * tf.reduce_mean(self.max_logvar) - 0.01 * tf.reduce_mean(self.min_logvar)
                                       + self.q_loss_importance * q_losses[i]
                                       for i in range(2)]

            self.optimizer = [optimizer(learning_rate=self.learning_rate) for _ in range(2)]

            current_scope = name
            trainable_vfun_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=current_scope)
            self.vfun_params = OrderedDict([(remove_scope_from_name(var.name, current_scope), var) for var in trainable_vfun_vars])
            keys = []
            for key in self.vfun_params.keys():
                if key[:6] == "policy" or key[:5] == "q_fun":
                    keys.append(key)
            for key in keys:
                self.vfun_params.pop(key)
            vars = list(self.vfun_params.values())
            self.train_op = [self.optimizer[i].minimize(self.loss[i], var_list = vars) for i in range(2)]
            # tensor_utils
            self.f_delta_pred = compile_function([self.obs_ph, self.act_ph], self.delta_pred)
            self.f_var_pred = compile_function([self.obs_ph, self.act_ph], self.var_pred)

        """ computation graph for inference where each of the models receives a different batch"""
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # placeholders
            self.obs_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.rew_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None,))
            self.done_model_batches_stack_ph = tf.placeholder(tf.bool, shape=(None,))

            # split stack into the batches for each model --> assume each model receives a batch of the same size
            self.obs_model_batches = tf.split(self.obs_model_batches_stack_ph, self.num_models, axis=0)
            self.act_model_batches = tf.split(self.act_model_batches_stack_ph, self.num_models, axis=0)
            self.delta_model_batches = tf.split(self.delta_model_batches_stack_ph, self.num_models, axis=0)
            self.rew_model_batches = tf.split(self.rew_model_batches_stack_ph, self.num_models, axis=0)
            self.done_model_batches = tf.split(self.done_model_batches_stack_ph, self.num_models, axis=0)

            # reuse previously created MLP but each model receives its own batch
            delta_preds = []
            var_preds = []
            self.obs_next_pred = []
            self.loss_model_batches = []
            self.train_op_model_batches = []
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):
                    # concatenate action and observation --> NN input
                    nn_input = tf.concat([self.obs_model_batches[i], self.act_model_batches[i]], axis=1)
                    mlp = MLP(name+'/model_{}'.format(i),
                              output_dim=2 * obs_space_dims,
                              hidden_sizes=hidden_sizes,
                              hidden_nonlinearity=hidden_nonlinearity,
                              output_nonlinearity=output_nonlinearity,
                              input_var=nn_input,
                              input_dim=obs_space_dims+action_space_dims,
                              weight_normalization=weight_normalization)

                mean, logvar = tf.split(mlp.output_var, 2, axis=-1)
                logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
                logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
                var = tf.exp(logvar)
                inv_var = tf.exp(-logvar)

                loss = tf.reduce_mean(tf.square(self.delta_model_batches[i] - mean) * inv_var + logvar)
                loss += (0.01 * tf.reduce_mean(self.max_logvar) - 0.01 * tf.reduce_mean(self.min_logvar))

                delta_preds.append(mean)
                var_preds.append(var)
                self.loss_model_batches.append(loss)
                self.train_op_model_batches.append(optimizer(learning_rate=self.learning_rate).minimize(loss))

            self.delta_pred_model_batches_stack = tf.concat(delta_preds, axis=0) # shape: (batch_size_per_model*num_models, ndim_obs)
            self.var_pred_model_batches_stack = tf.concat(var_preds, axis=0)

            # tensor_utils
            self.f_delta_pred_model_batches = compile_function([self.obs_model_batches_stack_ph,
                                                                self.act_model_batches_stack_ph],
                                                                self.delta_pred_model_batches_stack)

            self.f_var_pred_model_batches = compile_function([self.obs_model_batches_stack_ph,
                                                                self.act_model_batches_stack_ph],
                                                                self.var_pred_model_batches_stack)

        self._networks = mlps

    def predict_sym(self, obs_ph, act_ph, shuffle=True):
        """
        Same batch fed into all models. Randomly output one of the predictions for each observation.
        :param obs_ph: (batch_size, obs_space_dims)
        :param act_ph: (batch_size, act_space_dims)
        :return: (batch_size, obs_space_dims)
        """
        original_obs = obs_ph
        # shuffle
        if shuffle:
            perm = tf.range(0, limit=tf.shape(obs_ph)[0], dtype=tf.int32)
            perm = tf.random.shuffle(perm)
            obs_ph, act_ph = tf.gather(obs_ph, perm), tf.gather(act_ph, perm)
        obs_ph, act_ph = tf.split(obs_ph, self.num_models, axis=0), tf.split(act_ph, self.num_models, axis=0)

        delta_preds = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for i in range(self.num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=tf.AUTO_REUSE):
                    assert self.normalize_input
                    in_obs_var = (obs_ph[i] - self._mean_obs_var[i])/(self._std_obs_var[i] + 1e-8)
                    in_act_var = (act_ph[i] - self._mean_act_var[i]) / (self._std_act_var[i] + 1e-8)
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
                delta_pred = mean + tf.random.normal(shape=tf.shape(mean)) * tf.exp(logvar)
                # denormalize
                delta_pred = delta_pred * self._std_delta_var[i] + self._mean_delta_var[i]
                delta_preds.append(delta_pred)

        delta_preds = tf.concat(delta_preds, axis=0)

        # unshuffle
        if shuffle:
            perm_inv = tf.invert_permutation(perm)
            # next_obs = clip(obs + delta_pred
            next_obs = original_obs + tf.gather(delta_preds, perm_inv)
        else:
            next_obs = original_obs + delta_preds
        next_obs = tf.clip_by_value(next_obs, -1e2, 1e2)
        return next_obs

    def predict(self, obs, act, pred_type='rand', deterministic=False):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param pred_type:  prediction type
                   - rand: choose one of the models randomly
                   - mean: mean prediction of all models
                   - all: returns the prediction of all the models
                   - (int): returns the prediction of the i-th model
        :return: pred_obs_next: predicted batch of next observations -
                                shape:  (n_samples, ndim_obs) - in case of 'rand' and 'mean' mode
                                        (n_samples, ndim_obs, n_models) - in case of 'all' mode
        """
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        obs, act = [obs for _ in range(self.num_models)], [act for _ in range(self.num_models)]
        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)
            delta = np.array(self.f_delta_pred(obs, act))
            var = np.array(self.f_var_pred(obs, act))
            # print(obs, act, delta, var)
            if not deterministic:
                delta = np.random.normal(delta, np.sqrt(var))
            delta = self._denormalize_data(delta)
        else:
            obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)
            delta = np.array(self.f_delta_pred(obs, act))
            var = np.array(self.f_var_pred(obs, act))
            if not deterministic:
                delta = np.random.normal(delta, np.sqrt(var))

        assert delta.ndim == 3

        delta = np.clip(delta, -1e2, 1e2)

        pred_obs = obs_original[:, :, None] + delta

        batch_size = delta.shape[0]
        if pred_type == 'rand':
            # randomly selecting the prediction of one model in each row
            idx = np.random.randint(0, self.num_models, size=batch_size)
            pred_obs = np.stack([pred_obs[row, :, model_id] for row, model_id in enumerate(idx)], axis=0)
        elif pred_type == 'mean':
            pred_obs = np.mean(pred_obs, axis=2)
        elif pred_type == 'all':
            pass
        elif type(pred_type) is int:
            assert 0 <= pred_type < self.num_models
            pred_obs = pred_obs[:, :, pred_type]
        else:
            NotImplementedError('pred_type must be one of [rand, mean, all]')
        return pred_obs

    def _create_assign_ph(self):
        self._min_log_var_ph = tf.placeholder(tf.float32, shape=[1, self.obs_space_dims], name="min_logvar_ph")
        self._max_log_var_ph = tf.placeholder(tf.float32, shape=[1, self.obs_space_dims], name="max_logvar_ph")
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
