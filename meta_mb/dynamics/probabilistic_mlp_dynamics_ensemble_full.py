from meta_mb.dynamics.layers import MLP
import tensorflow as tf
import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.dynamics.mlp_dynamics_ensemble_full import MLPDynamicsEnsembleFull
from pdb import set_trace as st
from collections import OrderedDict
from meta_mb.utils.utils import remove_scope_from_name


class ProbMLPDynamicsEnsembleFull(MLPDynamicsEnsembleFull):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 discount = 1,
                 Qs=None,
                 Q_targets=None,
                 policy=None,
                 reward_scale=1,
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
                 q_loss_importance=0,
                 T=0,
                 buffer_size=50000,
                 type=1,
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

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]
        self.timesteps_counter = 0
        self.used_timesteps_counter = 0

        """added"""
        self.env = env
        if type == 2:
            self.Qs = Qs
            if Q_targets is None:
                self.Q_targets = Qs
            else:
                self.Q_targets = Q_targets
            self.policy = policy
            self.reward_scale = reward_scale
            self.discount = discount
            self.T = T
            self.discount=discount
            self.q_loss_importance = q_loss_importance
            log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
            alpha = tf.exp(log_alpha)
            self.log_alpha = log_alpha
            self.alpha = alpha

        hidden_nonlinearity = self._activations[hidden_nonlinearity]
        output_nonlinearity = self._activations[output_nonlinearity]
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        """ computation graph for training and simple inference """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self._create_stats_vars()

            self.max_logvar = tf.Variable(np.ones([1, obs_space_dims+1]) * max_logvar, dtype=tf.float32,
                                          trainable=True,
                                          name="max_logvar")
            self.min_logvar = tf.Variable(np.ones([1, obs_space_dims+1]) * min_logvar, dtype=tf.float32,
                                          trainable=True,
                                          name="min_logvar")
            self._create_assign_ph()

            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims+1))

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
                              output_dim=2 * (obs_space_dims+1),
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
            if type == 2:
                obs = self.obs_ph
                dist_info_sym = self.policy.distribution_info_sym(obs)
                actions, _ = self.policy.distribution.sample_sym(dist_info_sym)
                for t in range(self.T+1):
                    next_observation, _ = self.predict_sym(obs, actions)
                    dist_info_sym = self.policy.distribution_info_sym(next_observation)
                    next_actions_var, _ = self.policy.distribution.sample_sym(dist_info_sym)
                    rewards = self.env.tf_reward(obs, actions, next_observation)
                    rewards = tf.expand_dims(rewards, axis=-1)
                    dones_next = tf.cast(self.env.tf_termination_fn(obs, actions, next_observation), rewards.dtype)
                    if t == 0 :
                        reward_values = (self.discount**(t)) * self.reward_scale * rewards
                    else:
                        reward_values = (self.discount**(t)) * self.reward_scale * rewards * (1 - dones) + reward_values
                        input_q_fun = tf.concat([next_observation, next_actions_var], axis=-1)
                    dones = dones_next
                    obs, actions = next_observation, next_actions_var
                next_q_values = [(self.discount ** (self.T + 1)) * (1- dones) * Q.value_sym(input_var=input_q_fun) for Q in self.Qs]
                q_values = [reward_values + value for value in next_q_values]

                # define loss and train_op
                input_q_fun = tf.concat([(self.obs_ph-self._mean_obs_ph)/self._std_obs_ph+ 1e-8, (self.act_ph-self._mean_act_ph)/self._std_act_ph + 1e-8], axis=-1)

                q_values_var = [Q.value_sym(input_var=input_q_fun) for Q in self.Qs]

                q_losses = [self.q_loss_importance * tf.losses.mean_squared_error(labels=q_values[k], predictions=q_values_var[k], weights=0.5)
                                            for k in range(2)]
                # q_losses = [loss/tf.math.reduce_std(loss) + 1e-8 for loss in q_losses]

                self.loss = [tf.reduce_mean(tf.square(self.delta_ph[:, :, None] - self.delta_pred) * self.invar_pred
                                           + self.logvar_pred)
                                           + 0.01 * tf.reduce_mean(self.max_logvar) - 0.01 * tf.reduce_mean(self.min_logvar)
                                           + self.q_loss_importance * q_losses[k]
                                           for k in range(2)]

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
                self.train_op = [self.optimizer[k].minimize(self.loss[k], var_list = vars) for k in range(2)]
            else:
                self.loss = tf.reduce_mean(tf.square(self.delta_ph[:, :, None] - self.delta_pred) * self.invar_pred
                                           + self.logvar_pred)
                self.loss += 0.01 * tf.reduce_mean(self.max_logvar) - 0.01 * tf.reduce_mean(self.min_logvar)
                self.optimizer = optimizer(learning_rate=self.learning_rate)
                self.train_op = self.optimizer.minimize(self.loss)

            # tensor_utils
            self.f_delta_pred = compile_function([self.obs_ph, self.act_ph], self.delta_pred)
            self.f_var_pred = compile_function([self.obs_ph, self.act_ph], self.var_pred)

        """ computation graph for inference where each of the models receives a different batch"""
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # placeholders
            self.obs_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims+1))

            # split stack into the batches for each model --> assume each model receives a batch of the same size
            self.obs_model_batches = tf.split(self.obs_model_batches_stack_ph, self.num_models, axis=0)
            self.act_model_batches = tf.split(self.act_model_batches_stack_ph, self.num_models, axis=0)
            self.delta_model_batches = tf.split(self.delta_model_batches_stack_ph, self.num_models, axis=0)

            # reuse previously created MLP but each model receives its own batch
            delta_preds = []
            var_preds = []
            self.obs_next_pred = []
            self.loss_model_batches = []
            self.train_op_model_batches = []
            if type == 2:
                obs = self.obs_model_batches_stack_ph
                dist_info_sym = self.policy.distribution_info_sym(obs)
                actions, _ = self.policy.distribution.sample_sym(dist_info_sym)
                for t in range(self.T+1):
                    next_observation, _ = self.predict_sym(obs, actions)
                    dist_info_sym = self.policy.distribution_info_sym(next_observation)
                    next_actions_var, _ = self.policy.distribution.sample_sym(dist_info_sym)
                    rewards = self.env.tf_reward(obs, actions, next_observation)
                    rewards = tf.expand_dims(rewards, axis=-1)
                    dones_next = tf.cast(self.env.tf_termination_fn(obs, actions, next_observation), rewards.dtype)
                    if t == 0 :
                        reward_values = (self.discount**(t)) * self.reward_scale * rewards
                    else:
                        reward_values = (self.discount**(t)) * self.reward_scale * rewards * (1 - dones) + reward_values
                    dones = dones_next
                    obs, actions = next_observation, next_actions_var
                input_q_fun = tf.concat([next_observation, next_actions_var], axis=-1)
                next_q_values = [(self.discount ** (self.T + 1)) * (1- dones) * Q.value_sym(input_var=input_q_fun) for Q in self.Qs]
                q_values = [reward_values + value for value in next_q_values]
                q_values = [tf.split(q_value, self.num_models, axis = 0) for q_value in q_values]
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):
                    # concatenate action and observation --> NN input
                    nn_input = tf.concat([self.obs_model_batches[i], self.act_model_batches[i]], axis=1)
                    mlp = MLP(name+'/model_{}'.format(i),
                              output_dim=2 * (obs_space_dims+1),
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

                if type == 2:
                    # define loss and train_op
                    # normalize
                    in_obs_var = (self.obs_model_batches[i] - self._mean_obs_var[i])/(self._std_obs_var[i] + 1e-8)
                    in_act_var = (self.act_model_batches[i] - self._mean_act_var[i]) / (self._std_act_var[i] + 1e-8)
                    input_q_fun = tf.concat([in_obs_var, in_act_var], axis = -1)
                    q_values_var = [Q.value_sym(input_var=input_q_fun) for Q in self.Qs]

                    q_losses = [self.q_loss_importance * tf.losses.mean_squared_error(labels=q_values[k][i], predictions=q_values_var[k], weights=0.5)
                                                for k in range(2)]

                    # q_losses = [loss/tf.math.reduce_std(loss) for loss in q_losses]

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
                    loss = tf.reduce_mean(tf.square(self.delta_model_batches[i] - mean) * inv_var + logvar)
                    loss += (0.01 * tf.reduce_mean(self.max_logvar) - 0.01 * tf.reduce_mean(self.min_logvar))
                    loss += tf.reduce_min(q_losses, axis = 0)

                    self.train_op_model_batches.append(optimizer(learning_rate=self.learning_rate).minimize(loss, var_list = vars))

                    self.loss_model_batches.append(loss)
                    # self.train_op_model_batches.append(optimizer(learning_rate=self.learning_rate).minimize(loss))
                else:
                # define loss and train_op
                    loss = tf.reduce_mean(tf.square(self.delta_model_batches[i] - mean) * inv_var + logvar)
                    loss += (0.01 * tf.reduce_mean(self.max_logvar) - 0.01 * tf.reduce_mean(self.min_logvar))

                    self.loss_model_batches.append(loss)
                    self.train_op_model_batches.append(optimizer(learning_rate=self.learning_rate).minimize(loss))
                delta_preds.append(mean)
                var_preds.append(var)

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
                              output_dim=2 * (self.obs_space_dims+1),
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


        if shuffle:
            perm_inv = tf.invert_permutation(perm)
            original = tf.gather(delta_preds, perm_inv)
            pred_obs = tf.clip_by_value(original_obs + original[:, :self.obs_space_dims], -1e2, 1e2)
            pred_rewards = original[:, self.obs_space_dims:]
        else:
            pred_obs = tf.clip_by_value(original_obs + delta_preds[:, :self.obs_space_dims], -1e2, 1e2)
            pred_rewards = delta_preds[:, self.obs_space_dims:]

        return pred_obs, pred_rewards



    def predict(self, obs, act, pred_type='rand', deterministic=False, return_infos=False):
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

        pred_obs = obs_original[:, :, None] + delta[:, :self.obs_space_dims]
        predictions = np.concatenate([pred_obs, delta[:, self.obs_space_dims:]], axis = -2)

        if return_infos:  # info across all models in the ensemble
            agent_infos = [
                dict(mean=mean, std=std) for mean, std in zip(np.mean(pred_obs, axis=2), np.std(pred_obs, axis=2))
            ]

        if pred_type == 'rand':
            batch_size = delta.shape[0]
            # randomly selecting the prediction of one model in each row
            idx = np.random.randint(0, self.num_models, size=batch_size)
            predictions = np.stack([predictions[row, :, model_id] for row, model_id in enumerate(idx)], axis=0)
        elif pred_type == 'mean':
            predictions = np.mean(predictions, axis=2)
        elif pred_type == 'all':
            pass
        elif type(pred_type) is int:
            assert 0 <= pred_type < self.num_models
            predictions = predictions[:, :, pred_type]
        else:
            NotImplementedError('pred_type must be one of [rand, mean, all]')

        pred_obs = predictions[:, :self.obs_space_dims]
        rewards = predictions[:, self.obs_space_dims:]
        if return_infos:
            return pred_obs, rewards, agent_infos
        return pred_obs, rewards


    def _create_assign_ph(self):
        self._min_log_var_ph = tf.placeholder(tf.float32, shape=[1, self.obs_space_dims+1], name="min_logvar_ph")
        self._max_log_var_ph = tf.placeholder(tf.float32, shape=[1, self.obs_space_dims+1], name="max_logvar_ph")
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
