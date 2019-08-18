from meta_mb.dynamics.layers import MLP
import tensorflow as tf
import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.utils import tf_normalize, tf_denormalize


class ProbMLPDynamicsEnsemble(MLPDynamicsEnsemble):
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
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 buffer_size=50000,
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


        self.hidden_nonlinearity = hidden_nonlinearity = self._activations[hidden_nonlinearity]
        self.output_nonlinearity = output_nonlinearity = self._activations[output_nonlinearity]

        # NO NORNALIZATION INVOLVED
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

            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

            # concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)
            nn_input_split = tf.split(self.nn_input, self.num_models, axis=0)

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
                              input_var=nn_input_split[i],
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

            # define loss and train_op
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
            self.delta_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

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
                              weight_normalization=weight_normalization)  # FIXME: this mlp is not in self.networks?

                mean, logvar = tf.split(mlp.output_var, 2, axis=-1)
                logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
                logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
                var = tf.exp(logvar)
                inv_var = tf.exp(-logvar)

                delta_preds.append(mean)
                var_preds.append(var)

                # define loss and train_op
                loss = tf.reduce_mean(tf.square(self.delta_model_batches[i] - mean) * inv_var + logvar)
                loss += 0.01 * tf.reduce_mean(self.max_logvar) - 0.01 * tf.reduce_mean(self.min_logvar)
                self.loss_model_batches.append(loss)
                optimizer_model_batches = optimizer(learning_rate=self.learning_rate)
                self.train_op_model_batches.append(optimizer_model_batches.minimize(loss))

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

    def predict_sym(self, obs_ph, act_ph, pred_type, deterministic):
        """
        Same batch fed into all models. Randomly output one of the predictions for each observation.
        :param obs_ph: (batch_size, obs_space_dims)
        :param act_ph: (batch_size, act_space_dims)
        :return: (batch_size, obs_space_dims)
        """
        if pred_type == 'rand':
            # shuffle and split
            perm = tf.range(0, limit=tf.shape(obs_ph)[0], dtype=tf.int32)
            perm = tf.random.shuffle(perm)
            obs_ph_perm, act_ph_perm = tf.gather(obs_ph, perm), tf.gather(act_ph, perm)

            # split into num_models of batches and feed into each model
            pred_obs_perm = self.predict_batches_sym(obs_ph_perm, act_ph_perm, deterministic=deterministic)

            # unshuffle
            perm_inv = tf.invert_permutation(perm)
            pred_obs = tf.gather(pred_obs_perm, perm_inv)

            return pred_obs

        delta_preds = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for i in range(self.num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):
                    assert self.normalize_input
                    in_obs_var = tf_normalize(obs_ph, mean=self._mean_obs_var[i], std=self._std_obs_var[i])
                    in_act_var = tf_normalize(act_ph, mean=self._mean_act_var[i], std=self._std_act_var[i])
                    input_var = tf.concat([in_obs_var, in_act_var], axis=1)
                    mlp = MLP(self.name+'/model_{}'.format(i),
                              output_dim=2 * self.obs_space_dims,
                              hidden_sizes=self.hidden_sizes,
                              hidden_nonlinearity=self.hidden_nonlinearity,
                              output_nonlinearity=self.output_nonlinearity,
                              input_var=input_var,
                              input_dim=self.obs_space_dims + self.action_space_dims,
                              )

                    delta_pred, logvar = tf.split(mlp.output_var, 2, axis=-1)
                    logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
                    logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
                    if not deterministic:
                        delta_pred = delta_pred + tf.random.normal(shape=tf.shape(delta_pred)) * tf.exp(0.5*logvar)
                    delta_pred = tf_denormalize(delta_pred, mean=self._mean_delta_var[i], std=self._std_delta_var[i])

                    delta_preds.append(delta_pred)

        delta_preds = tf.stack(delta_preds, axis=2)  # (batch_size, obs_dims, num_models)
        pred_obs = tf.expand_dims(obs_ph, axis=2) + delta_preds

        if pred_type == 'all':
            pass
        elif pred_type == 'mean':
            pred_obs = tf.reduce_mean(pred_obs, axis=2)
        elif type(pred_type) is int:
            assert 0 <= pred_type < self.num_models
            pred_obs = pred_obs[:, :, pred_type]
        else:
            NotImplementedError('pred_type must be one of [rand, mean, all]')

        return tf.clip_by_value(pred_obs, -1e2, 1e2)

    def predict_batches_sym(self, obs_ph, act_ph, deterministic):
        """

        :param obs_ph: (batch_size, obs_space_dims)
        :param act_ph: (batch_size, act_space_dims)
        :return: (batch_size, obs_space_dims)
        """
        original_obs = obs_ph

        # split
        obs_ph, act_ph = tf.split(obs_ph, self.num_models, axis=0), tf.split(act_ph, self.num_models, axis=0)

        delta_preds = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for i in range(self.num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):
                    assert self.normalize_input
                    in_obs_var = tf_normalize(obs_ph[i], mean=self._mean_obs_var[i], std=self._std_obs_var[i])
                    in_act_var = tf_normalize(act_ph[i], mean=self._mean_act_var[i], std=self._std_act_var[i])

                    input_var = tf.concat([in_obs_var, in_act_var], axis=1)
                    mlp = MLP(self.name+'/model_{}'.format(i),
                              output_dim=2 * self.obs_space_dims,
                              hidden_sizes=self.hidden_sizes,
                              hidden_nonlinearity=self.hidden_nonlinearity,
                              output_nonlinearity=self.output_nonlinearity,
                              input_var=input_var,
                              input_dim=self.obs_space_dims + self.action_space_dims,
                              )

                    delta_pred, logvar = tf.split(mlp.output_var, 2, axis=-1)
                    logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
                    logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
                    if not deterministic:
                        delta_pred = delta_pred + tf.random.normal(shape=tf.shape(delta_pred)) * tf.exp(0.5*logvar)
                    delta_pred = tf_denormalize(delta_pred, mean=self._mean_delta_var[i], std=self._std_delta_var[i])

                    delta_preds.append(delta_pred)

        delta_preds = tf.concat(delta_preds, axis=0)
        pred_obs = original_obs + delta_preds

        return tf.clip_by_value(pred_obs, -1e2, 1e2)

    def predict(self, obs, act, pred_type, deterministic):
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

        if pred_type == 'rand' and obs.shape[0] % self.num_models == 0:
            batch_size = obs.shape[0]
            perm = np.random.permutation(batch_size)
            obs_perm, act_perm = obs[perm], act[perm]

            pred_obs_perm = self.predict_batches(obs_perm, act_perm, deterministic=deterministic)

            perm_inv = np.empty(batch_size, dtype=int)
            perm_inv[perm] = np.arange(batch_size)
            pred_obs = pred_obs_perm[perm_inv]

            return pred_obs

        obs_original = obs

        if self.normalize_input:
            # repeat obs, act to be num_models identical batches, normalize, and concatenate
            obs, act = [obs for _ in range(self.num_models)], [act for _ in range(self.num_models)]
            obs, act = self._normalize_data(obs, act)
            obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)

            delta = np.array(self.f_delta_pred(obs, act))
            var = np.array(self.f_var_pred(obs, act))

            if not deterministic:
                delta = np.random.normal(delta, np.sqrt(var))

            delta = self._denormalize_data(delta)
        else:
            obs, act = [obs for _ in range(self.num_models)], [act for _ in range(self.num_models)]
            obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)

            delta = np.array(self.f_delta_pred(obs, act))
            var = np.array(self.f_var_pred(obs, act))
            if not deterministic:
                delta = np.random.normal(delta, np.sqrt(var))

        assert delta.ndim == 3

        pred_obs = obs_original[:, :, None] + delta

        if pred_type == 'mean':
            pred_obs = np.mean(pred_obs, axis=2)
        elif pred_type == 'all':
            pass
        elif pred_type == 'rand':
            # randomly selecting the prediction of one model in each row
            idx = np.random.randint(0, self.num_models, size=pred_obs.shape[0])
            pred_obs = np.stack([pred_obs[row, :, model_id] for row, model_id in enumerate(idx)], axis=0)
        else:
            NotImplementedError('pred_type must be one of [rand, mean, all, (int)]')

        return np.clip(pred_obs, -1e2, 1e2)

    def predict_batches(self, obs_batches, act_batches, deterministic):
        """
            Predict the batch of next observations for each model given the batch of current observations and actions for each model
            :param obs_batches: observation batches for each model concatenated along axis 0 - numpy array of shape (batch_size_per_model * num_models, ndim_obs)
            :param act_batches: action batches for each model concatenated along axis 0 - numpy array of shape (batch_size_per_model * num_models, ndim_act)
            :return: pred_obs_next_batch: predicted batch of next observations -
                                    shape:  (batch_size_per_model * num_models, ndim_obs)
        """
        assert obs_batches.shape[0] == act_batches.shape[0] and obs_batches.shape[0] % self.num_models == 0
        assert obs_batches.ndim == 2 and obs_batches.shape[1] == self.obs_space_dims
        assert act_batches.ndim == 2 and act_batches.shape[1] == self.action_space_dims

        obs_batches_original = obs_batches

        if self.normalize_input:
            # normalize
            obs_batches, act_batches = np.split(obs_batches, self.num_models), np.split(act_batches, self.num_models)
            obs_batches, act_batches = self._normalize_data(obs_batches, act_batches)
            obs_batches, act_batches = np.concatenate(obs_batches, axis=0), np.concatenate(act_batches, axis=0)

            # predict delta
            delta_batches = np.array(self.f_delta_pred_model_batches(obs_batches, act_batches))
            var_batches = np.array(self.f_var_pred_model_batches(obs_batches, act_batches))
            if not deterministic:
                delta_batches = np.random.normal(delta_batches, np.sqrt(var_batches))

            # denormalize
            delta_batches = np.array(np.split(delta_batches, self.num_models)).transpose((1, 2, 0))
            delta_batches = self._denormalize_data(delta_batches)
            delta_batches = np.concatenate(delta_batches.transpose((2, 0, 1)), axis=0)
        else:
            delta_batches = np.array(self.f_delta_pred(obs_batches, act_batches))
            var_batches = np.array(self.f_var_pred_model_batches(obs_batches, act_batches))
            if not deterministic:
                delta_batches = np.random.normal(delta_batches, np.sqrt(var_batches))

        assert delta_batches.ndim == 2

        pred_obs_batches = obs_batches_original + delta_batches
        assert pred_obs_batches.shape == obs_batches.shape

        return np.clip(pred_obs_batches, -1e2, 1e2)

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
