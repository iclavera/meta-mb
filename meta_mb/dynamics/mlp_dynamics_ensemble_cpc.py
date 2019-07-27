from meta_mb.dynamics.layers import MLP
import tensorflow as tf
import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.logger import logger
from meta_mb.dynamics.mlp_dynamics import MLPDynamicsModel
import time
from collections import OrderedDict
from meta_mb.dynamics.utils import normalize, denormalize, train_test_split

class MLPDynamicsEnsemble(MLPDynamicsModel):
    """
    Class for MLP continous dynamics model
    """

    def __init__(self,
                 name,
                 env,
                 num_stack,
                 encoder,
                 latent_dim,
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
                 buffer_size=500,
                 loss_str='MSE',
                 ):

        Serializable.quick_init(self, locals())

        max_logvar = 1
        min_logvar = 0.1

        self.num_stack = num_stack
        self.encoder = encoder
        self.latent_dim = latent_dim

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_models = num_models
        self.hidden_sizes = hidden_sizes
        self.name = name
        self._dataset = None
        self._train_idx = None
        self._buffer_size = buffer_size


        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape
        self.action_space_dims = action_space_dims = env.action_space.shape[0]
        self.timesteps_counter = 0
        self.used_timesteps_counter = 0

        self.hidden_nonlinearity = hidden_nonlinearity = self._activations[hidden_nonlinearity]
        self.output_nonlinearity = output_nonlinearity = self._activations[output_nonlinearity]


        """ computation graph for inference where each of the models receives a different batch"""
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            # self._create_stats_vars()
            # placeholders
            self.obs_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, num_stack, *obs_space_dims))
            self.act_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.next_obs_model_batches_stack_ph = tf.placeholder(tf.float32, shape=(None, *obs_space_dims))

            z = tf.stop_gradient(self.encoder(tf.reshape(self.obs_model_batches_stack_ph, shape=(-1, *obs_space_dims))))
            z = tf.reshape(z, shape=(-1, num_stack * latent_dim))
            #z = #tf.keras.layers.TimeDistributed(self.encoder)(self.obs_model_batches_stack_ph).output
            next_z = tf.stop_gradient(self.encoder(self.next_obs_model_batches_stack_ph))

            # split stack into the batches for each model --> assume each model receives a batch of the same size
            self.z_model_batches = tf.split(z, self.num_models, axis=0)
            self.act_model_batches = tf.split(self.act_model_batches_stack_ph, self.num_models, axis=0)
            self.next_z_model_batches = tf.split(next_z, self.num_models, axis=0)


            mlps = []
            next_z_preds = []
            self.obs_next_pred = []
            self.loss_model_batches = []
            self.train_op_model_batches = []
            for i in range(num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=tf.AUTO_REUSE):
                    # concatenate action and observation --> NN input
                    nn_input = tf.concat([self.z_model_batches[i], self.act_model_batches[i]], axis=1)
                    mlp = MLP(name + '/model_{}'.format(i),
                              output_dim=latent_dim,
                              hidden_sizes=hidden_sizes,
                              hidden_nonlinearity=hidden_nonlinearity,
                              output_nonlinearity=output_nonlinearity,
                              input_var=nn_input,
                              input_dim=latent_dim * num_stack + action_space_dims,
                              weight_normalization=weight_normalization)
                    mlps.append(mlp)

                next_z_preds.append(mlp.output_var)
                if loss_str == 'L2':
                    loss = tf.reduce_mean(tf.linalg.norm(self.next_z_model_batches[i] - mlp.output_var, axis=1))
                elif loss_str == 'MSE':
                    loss = tf.reduce_mean((self.next_z_model_batches[i] - mlp.output_var) ** 2)
                else:
                    raise NotImplementedError
                self.loss_model_batches.append(loss)
                self.train_op_model_batches.append(optimizer(learning_rate=self.learning_rate).minimize(loss))
            self.next_z_pred_model_batches_stack = tf.concat(next_z_preds,
                                                            axis=0)  # shape: (batch_size_per_model*num_models, ndim_obs)

            # tensor_utils
            self.f_delta_pred_model_batches = compile_function([self.obs_model_batches_stack_ph,
                                                                self.act_model_batches_stack_ph],
                                                               self.next_z_pred_model_batches_stack)

        self._networks = mlps
        # LayersPowered.__init__(self, [mlp.output_layer for mlp in mlps])

    def update_buffer(self, obs, act, obs_next, valid_split_ratio, check_init=True):
        """
        :param obs: shape N x T x img_size
        :param act: shape N x T x ac_dim
        :param obs_next: shape N x T x img_size
        """

        assert obs.ndim == 5 and obs.shape[2:] == self.obs_space_dims
        assert obs_next.ndim == 5 and obs_next.shape[2:] == self.obs_space_dims
        assert act.ndim == 3 and act.shape[2] == self.action_space_dims

        self.timesteps_counter += obs.shape[0]
        obs_seq = np.concatenate([obs, obs_next[:, -1:, :]], axis=1)
        # If case should be entered exactly once
        if check_init and self._dataset is None:
            self._dataset = dict(obs=obs_seq, act=act)
            self.update_train_idx(valid_split_ratio)

            # assert self.next_batch is None
            # self.next_batch, self.iterator = self._data_input_fn(self._dataset_train['obs'],
            #                                                      self._dataset_train['act'],
            #                                                      self._dataset_train['delta'],
            #                                                      batch_size=self.batch_size)
            # assert self.normalization is None
            # if self.normalize_input:
            #     self.compute_normalization(self._dataset_train['obs'],
            #                                self._dataset_train['act'],
            #                                self._dataset_train['delta'])
        else:
            n_new_samples = len(obs)
            n_max = self._buffer_size - n_new_samples

            self._dataset['obs'] = np.concatenate([self._dataset['obs'][-n_max:], obs_seq])
            self._dataset['act'] = np.concatenate([self._dataset['act'][-n_max:], act])

            self.update_train_idx(valid_split_ratio, n_new_samples=n_new_samples)

        logger.log('Model has dataset with size {}'.format(len(self._dataset['obs'])))

    def update_train_idx(self, valid_split_ratio, n_new_samples=0):
        if self._train_idx is None:
            self._train_idx = []
            dataset_size = len(self._dataset['obs'])
            train_size = int(dataset_size * (1 - valid_split_ratio))
            for _ in range(self.num_models):
                indices = np.random.choice(dataset_size, train_size, replace=False)
                train_idx = np.zeros(shape=dataset_size)
                train_idx[indices] = 1
                self._train_idx.append(train_idx)
        else:
            assert n_new_samples
            n_max = self._buffer_size - n_new_samples

            for i in range(self.num_models):
                old_train_idx = self._train_idx[i]

                train_size = int(self._dataset['obs'].shape[0] * (1 - valid_split_ratio)) - np.sum(old_train_idx[-n_max:]).astype(int)
                indices = np.random.choice(n_new_samples, train_size, replace=False)
                train_idx = np.zeros(shape=n_new_samples)
                train_idx[indices] = 1
                self._train_idx[i] = np.concatenate([old_train_idx[-n_max:], train_idx])

    def generate_batch(self, test=False):
        data_act, data_obs = self._dataset['act'], self._dataset['obs']
        ret_obs = []
        ret_next_obs = []
        ret_actions = []
        data_obs = np.concatenate([np.zeros((data_obs.shape[0], self.num_stack - 1, *data_obs.shape[2:])), data_obs], axis=1)
        obs_stack = np.stack([data_obs[:, offset: data_obs.shape[1] + offset - self.num_stack + 1]
                              for offset in range(self.num_stack)], axis=2)[:, :-1]
        assert len(self._train_idx[0]) == len(data_act) == len(data_obs), 'the three are %d, %d, and %d respectively' \
                                                                          % (len(self._train_idx[0]),len(data_act),len(data_obs))

        if test:
            for i in range(self.num_models):
                select_idx = np.arange(data_act.shape[0])[(1 - self._train_idx[i]).astype(bool)]
                obs = np.concatenate(obs_stack[select_idx])
                actions = np.concatenate(data_act[select_idx],  axis=0)
                next_obs = np.concatenate(data_obs[select_idx][:, self.num_stack:], axis=0)

                ret_obs.append(obs)
                ret_actions.append(actions)
                ret_next_obs.append(next_obs)

        else:
            for i in range(self.num_models):
                select_idx = np.arange(data_act.shape[0])[self._train_idx[i].astype(bool)]
                obs = np.concatenate(obs_stack[select_idx])
                actions = np.concatenate(data_act[select_idx], axis=0)
                next_obs = np.concatenate(data_obs[select_idx][:, self.num_stack:], axis=0)

                assert obs.shape[0] == actions.shape[0] == next_obs.shape[0]
                batch_idx = np.random.choice(obs.shape[0], size=self.batch_size)

                ret_obs.append(obs[batch_idx])
                ret_actions.append(actions[batch_idx])
                ret_next_obs.append(next_obs[batch_idx])

        return np.concatenate(ret_obs), np.concatenate(ret_actions), np.concatenate(ret_next_obs)



    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False, prefix=''):
        """
        Fits the NN dynamics model
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after taking action - numpy array of shape (n_samples, ndim_obs)
        :param epochs: number of training epochs
        :param compute_normalization: boolean indicating whether normalization shall be (re-)computed given the data
        :param valid_split_ratio: relative size of validation split (float between 0.0 and 1.0)
        :param (boolean) whether to log training stats in tabular format
        :param verbose: logging verbosity
        """
        if valid_split_ratio is None:
            valid_split_ratio = self.valid_split_ratio

        if obs is not None:
            self.update_buffer(obs, act, obs_next, valid_split_ratio)

        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        sess = tf.get_default_session()

        valid_loss_rolling_average = None
        train_op_to_do = self.train_op_model_batches
        idx_to_remove = []
        epoch_times = []
        epochs_per_model = []

        """ ------- Looping over training epochs ------- """
        for epoch in range(epochs):

            dataset_size = self._dataset['obs'].shape[0]
            dataset_size_in_trans = dataset_size * self._dataset['obs'].shape[1]

            # preparations for recording training stats
            epoch_start_time = time.time()
            batch_losses = []

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            for _ in range(dataset_size_in_trans // self.batch_size):
                obs_batch_stack, act_batch_stack, next_obs_batch_stack = self.generate_batch()

                # run train op
                batch_loss_train_ops = sess.run(self.loss_model_batches + train_op_to_do,
                                               feed_dict={self.obs_model_batches_stack_ph: obs_batch_stack,
                                                          self.act_model_batches_stack_ph: act_batch_stack,
                                                          self.next_obs_model_batches_stack_ph: next_obs_batch_stack})

                batch_loss = np.array(batch_loss_train_ops[:self.num_models])
                batch_losses.append(batch_loss)

            """ ------- Calculating validation loss ------- """
            obs_test_stack, act_test_stack, next_obs_test_stack = self.generate_batch(test=True)

            # compute validation loss
            valid_loss = sess.run(self.loss_model_batches,
                                  feed_dict={self.obs_model_batches_stack_ph: obs_test_stack,
                                             self.act_model_batches_stack_ph: act_test_stack,
                                             self.next_obs_model_batches_stack_ph: next_obs_test_stack})
            valid_loss = np.array(valid_loss)
            if valid_loss_rolling_average is None:
                valid_loss_rolling_average = 1.5 * valid_loss  # set initial rolling to a higher value avoid too early stopping
                valid_loss_rolling_average_prev = 2.0 * valid_loss
                for i in range(len(valid_loss)):
                    if valid_loss[i] < 0:
                        valid_loss_rolling_average[i] = valid_loss[i]/1.5  # set initial rolling to a higher value avoid too early stopping
                        valid_loss_rolling_average_prev[i] = valid_loss[i]/2.0

            valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average \
                                         + (1.0-rolling_average_persitency)*valid_loss

            if verbose:
                str_mean_batch_losses = ' '.join(['%.4f'%x for x in np.mean(batch_losses, axis=0)])
                str_valid_loss = ' '.join(['%.4f'%x for x in valid_loss])
                str_valid_loss_rolling_averge = ' '.join(['%.4f'%x for x in valid_loss_rolling_average])
                logger.log(
                    "Training NNDynamicsModel - finished epoch %i --\n"
                    "train loss: %s\nvalid loss: %s\nvalid_loss_mov_avg: %s"
                    %(epoch, str_mean_batch_losses, str_valid_loss, str_valid_loss_rolling_averge)
                )


            for i in range(self.num_models):
                if (valid_loss_rolling_average_prev[i] < valid_loss_rolling_average[i] or epoch == epochs - 1) and i not in idx_to_remove:
                    idx_to_remove.append(i)
                    epochs_per_model.append(epoch)
                    if epoch < epochs - 1:
                        logger.log('At Epoch {}, stop model {} since its valid_loss_rolling_average decreased'.format(epoch, i))

            train_op_to_do = [op for idx, op in enumerate(self.train_op_model_batches) if idx not in idx_to_remove]

            if not idx_to_remove: epoch_times.append(time.time() - epoch_start_time) # only track epoch times while all models are trained

            if not train_op_to_do:
                if verbose and epoch < epochs - 1:
                    logger.log('Stopping all DynamicsEnsemble Training before reaching max_num_epochs')
                break
            valid_loss_rolling_average_prev = valid_loss_rolling_average

        """ ------- Tabular Logging ------- """
        if log_tabular:
            logger.logkv(prefix+'AvgModelEpochTime', np.mean(epoch_times))
            assert len(epochs_per_model) == self.num_models
            logger.logkv(prefix+'AvgEpochs', np.mean(epochs_per_model))
            logger.logkv(prefix+'StdEpochs', np.std(epochs_per_model))
            logger.logkv(prefix+'MaxEpochs', np.max(epochs_per_model))
            logger.logkv(prefix+'MinEpochs', np.min(epochs_per_model))
            logger.logkv(prefix+'AvgFinalTrainLoss', np.mean(batch_losses))
            logger.logkv(prefix+'AvgFinalValidLoss', np.mean(valid_loss))
            logger.logkv(prefix+'AvgFinalValidLossRoll', np.mean(valid_loss_rolling_average))

    def predict_sym(self, obs_ph, act_ph):
        """
        Same batch fed into all models. Randomly output one of the predictions for each observation.
        :param obs_ph: (batch_size, obs_space_dims)
        :param act_ph: (batch_size, act_space_dims)
        :return: (batch_size, obs_space_dims)
        """
        original_obs = obs_ph
        obs_ph = tf.reshape(original_obs, shape=(-1, self.num_stack * self.latent_dim))
        # input_image = (len(obs_ph.shape) > 2)
        # if input_image:
        #     original_obs = self.encoder(tf.reshape(obs_ph, shape=(-1, *self.obs_space_dims)))
        #     # TODO: check that this automatically reuses variables
        #     original_obs = tf.reshape(original_obs, shape=(-1, self.num_stack * self.latent_dim))

        # shuffle
        perm = tf.range(0, limit=tf.shape(original_obs)[0], dtype=tf.int32)
        perm = tf.random.shuffle(perm)

        z, act_ph = tf.gather(obs_ph, perm), tf.gather(act_ph, perm)
        z, act_ph = tf.split(z, self.num_models, axis=0), tf.split(act_ph, self.num_models, axis=0)

        next_z_preds = []
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            for i in range(self.num_models):
                with tf.variable_scope('model_{}'.format(i), reuse=True):
                    # assert self.normalize_input
                    # in_obs_var = (obs_ph[i] - self._mean_obs_var[i])/(self._std_obs_var[i] + 1e-8)
                    # in_act_var = (act_ph[i] - self._mean_act_var[i]) / (self._std_act_var[i] + 1e-8)
                    input_var = tf.concat([z[i], act_ph[i]], axis=1)
                    mlp = MLP(self.name+'/model_{}'.format(i),
                              output_dim=self.latent_dim,
                              hidden_sizes=self.hidden_sizes,
                              hidden_nonlinearity=self.hidden_nonlinearity,
                              output_nonlinearity=self.output_nonlinearity,
                              input_var=input_var,
                              input_dim=self.latent_dim * self.num_stack + self.action_space_dims,
                              )
                    # denormalize delta_pred
                    # delta_pred = mlp.output_var * self._std_delta_var[i] + self._mean_delta_var[i]
                    next_z_preds.append(mlp.output_var)

        next_z_preds = tf.concat(next_z_preds, axis=0)

        # unshuffle
        perm_inv = tf.invert_permutation(perm)
        next_z = tf.gather(next_z_preds, perm_inv)
        next_z = tf.clip_by_value(next_z, -1e2, 1e2)
        return tf.concat([original_obs[:, 1:], next_z[:, None, :]], axis = 1)


    def predict(self, obs, act, pred_type='rand', **kwargs):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param pred_type:  prediction type
                   - rand: choose one of the models randomly
                   - mean: mean prediction of all models
                   - all: returns the prediction of all the models
        :return: pred_obs_next: predicted batch of next observations -
                                shape:  (n_samples, ndim_obs) - in case of 'rand' and 'mean' mode
                                        (n_samples, ndim_obs, n_models) - in case of 'all' mode
        """
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        if self.normalize_input:
            obs, act = [obs for _ in range(self.num_models)], [act for _ in range(self.num_models)]
            obs, act = self._normalize_data(obs, act)
            obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)
            delta = np.array(self.f_delta_pred(obs, act))
            delta = self._denormalize_data(delta)

        else:
            obs, act = [obs for _ in range(self.num_models)], [act for _ in range(self.num_models)]
            obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)
            delta = np.array(self.f_delta_pred(obs, act))

        assert delta.ndim == 3

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
        else:
            NotImplementedError('pred_type must be one of [rand, mean, all]')

        pred_obs = np.clip(pred_obs, -1e2, 1e2)
        return pred_obs


    def predict_std(self, obs, act):
        """
        Calculates the std of predicted next observations among the models
        given the batch of current observations and actions
        1:param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :return: std_pred_obs: std of predicted next observatations - (n_samples, ndim_obs)
        """
        assert self.num_models > 1, "calculating the std requires at "
        pred_obs = self.predict(obs, act, pred_type='all')
        assert pred_obs.ndim == 3
        return np.std(pred_obs, axis=2)

    def reinit_model(self):
        sess = tf.get_default_session()
        if '_reinit_model_op' not in dir(self):
            self._reinit_model_op = [tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope=self.name+'/model_{}'.format(i))) for i in range(self.num_models)]
        sess.run(self._reinit_model_op)

    def _data_input_fn(self, obs_batches, act_batches, delta_batches, batch_size=500, buffer_size=5000):
        """ Takes in train data an creates an a symbolic nex_batch operator as well as an iterator object """

        assert len(obs_batches) == len(act_batches) == len(delta_batches)
        obs, act, delta = obs_batches[0], act_batches[0], delta_batches[0]
        assert obs.ndim == act.ndim == delta.ndim, "inputs must have 2 dims"
        assert obs.shape[0] == act.shape[0] == delta.shape[0], "inputs must have same length along axis 0"
        assert obs.shape[1] == delta.shape[1], "obs and obs_next must have same length along axis 1 "

        self.obs_batches_dataset_ph = [tf.placeholder(tf.float32, (None, obs.shape[1])) for _ in range(self.num_models)]
        self.act_batches_dataset_ph = [tf.placeholder(tf.float32, (None, act.shape[1])) for _ in range(self.num_models)]
        self.delta_batches_dataset_ph = [tf.placeholder(tf.float32, (None, delta.shape[1])) for _ in range(self.num_models)]

        dataset = tf.data.Dataset.from_tensor_slices(
            tuple(self.obs_batches_dataset_ph + self.act_batches_dataset_ph + self.delta_batches_dataset_ph)
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        return next_batch, iterator

    def compute_normalization(self, obs, act, delta):
        assert len(obs) == len(act) == len(delta) == self.num_models
        assert all([o.shape[0] == d.shape[0] == a.shape[0] for o, a, d in zip(obs, act, delta)])
        assert all([d.shape[1] == o.shape[1] for d, o in zip(obs, delta)])

        # store means and std in dict
        self.normalization = []
        feed_dict = {}
        for i in range(self.num_models):
            normalization = OrderedDict()
            normalization['obs'] = (np.mean(obs[i], axis=0), np.std(obs[i], axis=0))
            normalization['delta'] = (np.mean(delta[i], axis=0), np.std(delta[i], axis=0))
            normalization['act'] = (np.mean(act[i], axis=0), np.std(act[i], axis=0))
            self.normalization.append(normalization)
            feed_dict.update({self._mean_obs_ph[i]: self.normalization[i]['obs'][0],
                         self._std_obs_ph[i]: self.normalization[i]['obs'][1],
                         self._mean_act_ph[i]: self.normalization[i]['act'][0],
                         self._std_act_ph[i]: self.normalization[i]['act'][1],
                         self._mean_delta_ph[i]: self.normalization[i]['delta'][0],
                         self._std_delta_ph[i]: self.normalization[i]['delta'][1],
                         }
                           )
        sess = tf.get_default_session()
        sess.run(self._assignations, feed_dict=feed_dict)

    def _create_stats_vars(self):
        self._mean_obs_var, self._std_obs_var, self._mean_obs_ph, self._std_obs_ph = [], [], [], []
        self._mean_act_var, self._std_act_var, self._mean_act_ph, self._std_act_ph = [], [], [], []
        self._mean_delta_var, self._std_delta_var, self._mean_delta_ph, self._std_delta_ph = [], [], [], []
        self._assignations = []
        for i in range(self.num_models):
            self._mean_obs_var.append(tf.get_variable('mean_obs_%d' % i, shape=(self.obs_space_dims,),
                                                 dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False))
            self._std_obs_var.append(tf.get_variable('std_obs_%d' % i, shape=(self.obs_space_dims,),
                                                dtype=tf.float32, initializer=tf.ones_initializer, trainable=False))
            self._mean_act_var.append(tf.get_variable('mean_act_%d' % i, shape=(self.action_space_dims,),
                                                 dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False))
            self._std_act_var.append(tf.get_variable('std_act_%d' % i, shape=(self.action_space_dims,),
                                                dtype=tf.float32, initializer=tf.ones_initializer, trainable=False))
            self._mean_delta_var.append(tf.get_variable('mean_delta_%d' % i, shape=(self.obs_space_dims,),
                                                   dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False))
            self._std_delta_var.append(tf.get_variable('std_delta_%d' % i, shape=(self.obs_space_dims,),
                                                  dtype=tf.float32, initializer=tf.ones_initializer, trainable=False))

            self._mean_obs_ph.append(tf.placeholder(tf.float32, shape=(self.obs_space_dims,)))
            self._std_obs_ph.append(tf.placeholder(tf.float32, shape=(self.obs_space_dims,)))
            self._mean_act_ph.append(tf.placeholder(tf.float32, shape=(self.action_space_dims,)))
            self._std_act_ph.append(tf.placeholder(tf.float32, shape=(self.action_space_dims,)))
            self._mean_delta_ph.append(tf.placeholder(tf.float32, shape=(self.obs_space_dims,)))
            self._std_delta_ph.append(tf.placeholder(tf.float32, shape=(self.obs_space_dims,)))

            self._assignations.extend([tf.assign(self._mean_obs_var[i], self._mean_obs_ph[i]),
                                      tf.assign(self._std_obs_var[i], self._std_obs_ph[i]),
                                      tf.assign(self._mean_act_var[i], self._mean_act_ph[i]),
                                      tf.assign(self._std_act_var[i], self._std_act_ph[i]),
                                      tf.assign(self._mean_delta_var[i], self._mean_delta_ph[i]),
                                      tf.assign(self._std_delta_var[i], self._std_delta_ph[i]),
                                      ])

    def _normalize_data(self, obs, act, delta=None):
        assert len(obs) == len(act) == self.num_models
        assert self.normalization is not None
        norm_obses = []
        norm_acts = []
        norm_deltas = []
        for i in range(self.num_models):
            norm_obs = normalize(obs[i], self.normalization[i]['obs'][0], self.normalization[i]['obs'][1])
            norm_act = normalize(act[i], self.normalization[i]['act'][0], self.normalization[i]['act'][1])
            norm_obses.append(norm_obs)
            norm_acts.append(norm_act)
            if delta is not None:
                assert len(delta) == self.num_models
                norm_delta = normalize(delta[i], self.normalization[i]['delta'][0], self.normalization[i]['delta'][1])
                norm_deltas.append(norm_delta)

        if delta is not None:
            return norm_obses, norm_acts, norm_deltas

        return norm_obses, norm_acts

    def _denormalize_data(self, delta):
        assert delta.shape[-1] == self.num_models
        denorm_deltas = []
        for i in range(self.num_models):
            denorm_delta = denormalize(delta[..., i], self.normalization[i]['delta'][0], self.normalization[i]['delta'][1])
            denorm_deltas.append(denorm_delta)
        return np.stack(denorm_deltas, axis=-1)

    def get_shared_param_values(self): # to feed policy
        state = dict()
        state['normalization'] = self.normalization
        state['networks_params'] = [nn.get_param_values() for nn in self._networks]
        return state

    def set_shared_params(self, state):
        self.normalization = state['normalization']
        feed_dict = {}
        for i in range(self.num_models):
            feed_dict.update({
                self._mean_obs_ph[i]: self.normalization[i]['obs'][0],
                self._std_obs_ph[i]: self.normalization[i]['obs'][1],
                self._mean_act_ph[i]: self.normalization[i]['act'][0],
                self._std_act_ph[i]: self.normalization[i]['act'][1],
                self._mean_delta_ph[i]: self.normalization[i]['delta'][0],
                self._std_delta_ph[i]: self.normalization[i]['delta'][1],
            })
        sess = tf.get_default_session()
        sess.run(self._assignations, feed_dict=feed_dict)
        for i in range(len(self._networks)):
            self._networks[i].set_params(state['networks_params'][i])




