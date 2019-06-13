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
import multiprocessing

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
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 buffer_size=50000,
                 ):

        Serializable.quick_init(self, locals())

        max_logvar = 1
        min_logvar = 0.1

        self.normalization = None
        self.normalize_input = normalize_input
        self.next_batch = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_models = num_models
        self.name = name
        self._dataset_train = None
        self._dataset_test = None

        # determine dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]

        hidden_nonlinearity = self._activations[hidden_nonlinearity]
        output_nonlinearity = self._activations[output_nonlinearity]

        """ computation graph for training and simple inference """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, obs_space_dims))

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
            self.loss = tf.reduce_mean((self.delta_ph[:, :, None] - self.delta_pred)**2)
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
                loss = tf.reduce_mean((self.delta_model_batches[i] - mlp.output_var) ** 2)
                self.loss_model_batches.append(loss)
                self.train_op_model_batches.append(optimizer(learning_rate=self.learning_rate).minimize(loss))
            self.delta_pred_model_batches_stack = tf.concat(delta_preds, axis=0) # shape: (batch_size_per_model*num_models, ndim_obs)

            # tensor_utils
            self.f_delta_pred_model_batches = compile_function([self.obs_model_batches_stack_ph,
                                                                self.act_model_batches_stack_ph],
                                                                self.delta_pred_model_batches_stack)

        self._networks = mlps
        # LayersPowered.__init__(self, [mlp.output_layer for mlp in mlps])

    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False):
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

        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.get_default_session()

        # split into valid and test set
        obs_train_batches = []
        act_train_batches = []
        delta_train_batches = []
        obs_test_batches = []
        act_test_batches = []
        delta_test_batches = []

        delta = obs_next - obs
        for i in range(self.num_models):
            obs_train, act_train, delta_train, obs_test, act_test, delta_test = train_test_split(obs, act, delta,
                                                                                             test_split_ratio=valid_split_ratio)
            obs_train_batches.append(obs_train)
            act_train_batches.append(act_train)
            delta_train_batches.append(delta_train)
            obs_test_batches.append(obs_test)
            act_test_batches.append(act_test)
            delta_test_batches.append(delta_test)
            # create data queue
        if self._dataset_test is None:
            self._dataset_test = dict(obs=obs_test_batches, act=act_test_batches, delta=delta_test_batches)
            self._dataset_train = dict(obs=obs_train_batches, act=act_train_batches, delta=delta_train_batches)
        else:
            n_test_new_samples = len(obs_test_batches[0])
            n_max_test = self.buffer_size - n_test_new_samples
            n_train_new_samples = len(obs_train_batches[0])
            n_max_train = self.buffer_size - n_train_new_samples
            for i in range(self.num_models):
                self._dataset_test['obs'][i] = np.concatenate([self._dataset_test['obs'][i][-n_max_test:],
                                                               obs_test_batches[i]])
                self._dataset_test['act'][i] = np.concatenate([self._dataset_test['act'][i][-n_max_test:],
                                                               act_test_batches[i]])
                self._dataset_test['delta'][i] = np.concatenate([self._dataset_test['delta'][i][-n_max_test:],
                                                                 delta_test_batches[i]])

                self._dataset_train['obs'][i] = np.concatenate([self._dataset_train['obs'][i][-n_max_train:],
                                                                obs_train_batches[i]])
                self._dataset_train['act'][i] = np.concatenate([self._dataset_train['act'][i][-n_max_train:],
                                                                act_train_batches[i]])
                self._dataset_train['delta'][i] = np.concatenate(
                    [self._dataset_train['delta'][i][-n_max_train:], delta_train_batches[i]])

            print("Size Data Set:  ", self._dataset_train['delta'][0].shape[0])

        if self.next_batch is None:
            self.next_batch, self.iterator = self._data_input_fn(self._dataset_train['obs'],
                                                                 self._dataset_train['act'],
                                                                 self._dataset_train['delta'],
                                                                 batch_size=self.batch_size)

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(self._dataset_train['obs'],
                                       self._dataset_train['act'],
                                       self._dataset_train['delta'])
        if self.normalize_input:
            # normalize data
            obs_train, act_train, delta_train = self._normalize_data(self._dataset_train['obs'],
                                                                     self._dataset_train['act'],
                                                                     self._dataset_train['delta'])
        else:
            obs_train, act_train, delta_train = self._dataset_train['obs'], self._dataset_train['act'],\
                                                self._dataset_train['delta']

        valid_loss_rolling_average = None
        train_op_to_do = self.train_op_model_batches
        idx_to_remove = []
        epoch_times = []
        epochs_per_model = []

        """ ------- Looping over training epochs ------- """
        for epoch in range(epochs):

            # initialize data queue
            feed_dict = dict(
                list(zip(self.obs_batches_dataset_ph, obs_train)) +
                list(zip(self.act_batches_dataset_ph, act_train)) +
                list(zip(self.delta_batches_dataset_ph, delta_train))
            )
            sess.run(self.iterator.initializer, feed_dict=feed_dict)

            # preparations for recording training stats
            epoch_start_time = time.time()
            batch_losses = []

            """ ------- Looping through the shuffled and batched dataset for one epoch -------"""
            while True:
                try:
                    obs_act_delta = sess.run(self.next_batch)
                    obs_batch_stack = np.concatenate(obs_act_delta[:self.num_models], axis=0)
                    act_batch_stack = np.concatenate(obs_act_delta[self.num_models:2*self.num_models], axis=0)
                    delta_batch_stack = np.concatenate(obs_act_delta[2*self.num_models:], axis=0)

                    # run train op
                    batch_loss_train_ops = sess.run(self.loss_model_batches + train_op_to_do,
                                                   feed_dict={self.obs_model_batches_stack_ph: obs_batch_stack,
                                                              self.act_model_batches_stack_ph: act_batch_stack,
                                                              self.delta_model_batches_stack_ph: delta_batch_stack})

                    batch_loss = np.array(batch_loss_train_ops[:self.num_models])
                    batch_losses.append(batch_loss)

                except tf.errors.OutOfRangeError:
                    if self.normalize_input:
                        # normalize data
                        obs_test, act_test, delta_test = self._normalize_data(self._dataset_test['obs'],
                                                                              self._dataset_test['act'],
                                                                              self._dataset_test['delta'])

                    else:
                        obs_test, act_test, delta_test = self._dataset_test['obs'], self._dataset_test['act'], \
                                                         self._dataset_test['delta']

                    obs_test_stack = np.concatenate(obs_test, axis=0)
                    act_test_stack = np.concatenate(act_test, axis=0)
                    delta_test_stack = np.concatenate(delta_test, axis=0)

                    # compute validation loss
                    valid_loss = sess.run(self.loss_model_batches,
                                          feed_dict={self.obs_model_batches_stack_ph: obs_test_stack,
                                                     self.act_model_batches_stack_ph: act_test_stack,
                                                     self.delta_model_batches_stack_ph: delta_test_stack})
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
                        logger.log("Training NNDynamicsModel - finished epoch %i --"
                                   "train loss: %s  valid loss: %s  valid_loss_mov_avg: %s"
                                   %(epoch, str_mean_batch_losses, str_valid_loss, str_valid_loss_rolling_averge))
                    break

            for i in range(self.num_models):
                if (valid_loss_rolling_average_prev[i] < valid_loss_rolling_average[i] or epoch == epochs - 1) and i not in idx_to_remove:
                    idx_to_remove.append(i)
                    epochs_per_model.append(epoch)
                    if verbose:
                        logger.log('Stopping Training of Model %i since its valid_loss_rolling_average decreased'%i)

            train_op_to_do = [op for idx, op in enumerate(self.train_op_model_batches) if idx not in idx_to_remove]

            if not idx_to_remove: epoch_times.append(time.time() - epoch_start_time) # only track epoch times while all models are trained

            if not train_op_to_do:
                logger.log('Stopping DynamicsEnsemble Training since valid_loss_rolling_average decreased')
                break
            valid_loss_rolling_average_prev = valid_loss_rolling_average

        """ ------- Tabular Logging ------- """
        if log_tabular:
            logger.logkv('AvgModelEpochTime', np.mean(epoch_times))
            assert len(epochs_per_model) == self.num_models
            logger.logkv('AvgEpochs', np.mean(epochs_per_model))
            logger.logkv('StdEpochsl', np.std(epochs_per_model))
            logger.logkv('AvgFinalTrainLoss', np.mean(batch_losses))
            logger.logkv('AvgFinalValidLoss', np.mean(valid_loss))
            logger.logkv('AvgFinalValidLoss', np.mean(valid_loss_rolling_average))

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

    def predict_batches(self, obs_batches, act_batches, *args, **kwargs):
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
            # Normalize Input
            obs_batches, act_batches = np.split(obs_batches, self.num_models), np.split(act_batches, self.num_models)
            obs_batches, act_batches = self._normalize_data(obs_batches, act_batches)
            obs_batches, act_batches = np.concatenate(obs_batches, axis=0), np.concatenate(act_batches, axis=0)

            delta_batches = np.array(self.f_delta_pred_model_batches(obs_batches, act_batches))

            # Denormalize output
            delta_batches = np.array(np.split(delta_batches, self.num_models)).transpose((1, 2, 0))
            delta_batches = self._denormalize_data(delta_batches)
            delta_batches = np.concatenate(delta_batches.transpose((2, 0, 1)), axis=0)

        else:
            delta_batches = np.array(self.f_delta_pred(obs_batches, act_batches))

        assert delta_batches.ndim == 2

        delta_batches = np.clip(delta_batches, -1e2, 1e2)

        pred_obs_batches = obs_batches_original + delta_batches
        assert pred_obs_batches.shape == obs_batches.shape
        return pred_obs_batches

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
        for i in range(self.num_models):
            normalization = OrderedDict()
            normalization['obs'] = (np.mean(obs[i], axis=0), np.std(obs[i], axis=0))
            normalization['delta'] = (np.mean(delta[i], axis=0), np.std(delta[i], axis=0))
            normalization['act'] = (np.mean(act[i], axis=0), np.std(act[i], axis=0))
            self.normalization.append(normalization)

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




