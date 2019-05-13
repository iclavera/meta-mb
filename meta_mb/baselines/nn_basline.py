# from meta_mb.core import MLP
from meta_mb.dynamics.layers import MLP
from meta_mb.dynamics.utils import normalize, denormalize, train_test_split
import tensorflow as tf
import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.utils import compile_function
from meta_mb.logger import logger
from collections import OrderedDict


class MLPDynamicsModel(Serializable):
    """
    Class for MLP continous dynamics model
    """
    _activations = {
        None: tf.identity,
        "relu": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x)
    }

    def __init__(self,
                 name,
                 env,
                 hidden_sizes=(500, 500),
                 hidden_nonlinearity="tanh",
                 output_nonlinearity=None,
                 batch_size=500,
                 step_size=0.001,
                 weight_normalization=True,
                 normalize_input=True,
                 optimizer=tf.train.AdamOptimizer,
                 valid_split_ratio=0.2,
                 rolling_average_persitency=0.99,
                 buffer_size=100000,
                 ):

        Serializable.quick_init(self, locals())

        self.normalization = None
        self.normalize_input = normalize_input
        self.use_reward_model = False
        self.buffer_size = buffer_size
        self.name = name

        self._dataset_train = None
        self._dataset_test = None

        self.valid_split_ratio = valid_split_ratio
        self.rolling_average_persitency = rolling_average_persitency
        hidden_nonlinearity = self._activations[hidden_nonlinearity]
        output_nonlinearity = self._activations[output_nonlinearity]

        with tf.variable_scope(name):
            self.batch_size = batch_size
            self.step_size = step_size

            # determine dimensionality of state and action space
            self.obs_space_dims = env.observation_space.shape[0]
            self.action_space_dims = env.action_space.shape[0]

            # placeholders
            self.obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_space_dims))
            self.act_ph = tf.placeholder(tf.float32, shape=(None, self.action_space_dims))
            self.delta_ph = tf.placeholder(tf.float32, shape=(None, self.obs_space_dims))

            # concatenate action and observation --> NN input
            self.nn_input = tf.concat([self.obs_ph, self.act_ph], axis=1)

            # create MLP
            with tf.variable_scope('dynamics_model'):
                mlp = MLP(name,
                          output_dim=self.obs_space_dims,
                          hidden_sizes=hidden_sizes,
                          hidden_nonlinearity=hidden_nonlinearity,
                          output_nonlinearity=output_nonlinearity,
                          input_var=self.nn_input,
                          input_dim=self.obs_space_dims + self.action_space_dims,
                          weight_normalization=weight_normalization)

            self.delta_pred = mlp.output_var

            # define loss and train_op
            self.loss = tf.reduce_mean((self.delta_ph - self.delta_pred)**2)
            self.optimizer = optimizer(self.step_size)
            self.train_op = self.optimizer.minimize(self.loss)

            # tensor_utils
            self.f_delta_pred = compile_function([self.obs_ph, self.act_ph], self.delta_pred)

        self._networks = [mlp]
        # LayersPowered.__init__(self, [mlp.output_layer])

    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True, verbose=False, valid_split_ratio=None, rolling_average_persitency=None):
        """
        Fits the NN dynamics model
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after taking action - numpy array of shape (n_samples, ndim_obs)
        :param epochs: number of training epochs
        :param compute_normalization: boolean indicating whether normalization shall be (re-)computed given the data
        :param valid_split_ratio: relative size of validation split (float between 0.0 and 1.0)
        :param verbose: logging verbosity
        """
        assert obs.ndim == 2 and obs.shape[1]==self.obs_space_dims
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        if valid_split_ratio is None: valid_split_ratio = self.valid_split_ratio
        if rolling_average_persitency is None: rolling_average_persitency = self.rolling_average_persitency

        assert 1 > valid_split_ratio >= 0

        sess = tf.get_default_session()

        if (self.normalization is None or compute_normalization) and self.normalize_input:
            self.compute_normalization(obs, act, obs_next)

        if self.normalize_input:
            # normalize data
            obs, act, delta = self._normalize_data(obs, act, obs_next)
            assert obs.ndim == act.ndim == obs_next.ndim == 2
        else:
            delta = obs_next - obs

        # split into valid and test set

        obs_train, act_train, delta_train, obs_test, act_test, delta_test = train_test_split(obs, act, delta,
                                                                                             test_split_ratio=valid_split_ratio)

        if self._dataset_test is None:
            self._dataset_test = dict(obs=obs_test, act=act_test, delta=delta_test)
            self._dataset_train = dict(obs=obs_train, act=act_train, delta=delta_train)
        else:
            n_test_new_samples = len(obs_test)
            n_max_test = self.buffer_size - n_test_new_samples
            n_train_new_samples = len(obs_train)
            n_max_train = self.buffer_size - n_train_new_samples
            self._dataset_test['obs'] = np.concatenate([self._dataset_test['obs'][-n_max_test:], obs_test])
            self._dataset_test['act'] = np.concatenate([self._dataset_test['act'][-n_max_test:], act_test])
            self._dataset_test['delta'] = np.concatenate([self._dataset_test['delta'][-n_max_test:], delta_test])

            self._dataset_train['obs'] = np.concatenate([self._dataset_train['obs'][-n_max_train:], obs_train])
            self._dataset_train['act'] = np.concatenate([self._dataset_train['act'][-n_max_train:], act_train])
            self._dataset_train['delta'] = np.concatenate([self._dataset_train['delta'][-n_max_train:], delta_train])

        # create data queue
        if self.next_batch is None:
            self.next_batch, self.iterator = self._data_input_fn(self._dataset_train['obs'],
                                                                 self._dataset_train['act'],
                                                                 self._dataset_train['delta'],
                                                                 batch_size=self.batch_size,
                                                                 buffer_size=self.buffer_size)

        valid_loss_rolling_average = None

        # Training loop
        for epoch in range(epochs):

            # initialize data queue
            sess.run(self.iterator.initializer,
                     feed_dict={self.obs_dataset_ph: self._dataset_train['obs'],
                                self.act_dataset_ph: self._dataset_train['act'],
                                self.delta_dataset_ph: self._dataset_train['delta']})

            batch_losses = []
            while True:
                try:
                    obs_batch, act_batch, delta_batch = sess.run(self.next_batch)

                    # run train op
                    batch_loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.obs_ph: obs_batch,
                                                                         self.act_ph: act_batch,
                                                                         self.delta_ph: delta_batch})

                    batch_losses.append(batch_loss)

                except tf.errors.OutOfRangeError:
                    # compute validation loss
                    valid_loss = sess.run(self.loss, feed_dict={self.obs_ph: self._dataset_test['obs'],
                                                                self.act_ph: self._dataset_test['act'],
                                                                self.delta_ph: self._dataset_test['delta']})
                    if valid_loss_rolling_average is None:
                        valid_loss_rolling_average = 1.5 * valid_loss # set initial rolling to a higher value avoid too early stopping
                        valid_loss_rolling_average_prev = 2.0 * valid_loss

                    valid_loss_rolling_average = rolling_average_persitency*valid_loss_rolling_average + (1.0-rolling_average_persitency)*valid_loss

                    if verbose:
                        logger.log("Training NNDynamicsModel - finished epoch %i -- train loss: %.4f  valid loss: %.4f  valid_loss_mov_avg: %.4f"%(epoch, float(np.mean(batch_losses)), valid_loss, valid_loss_rolling_average))
                    break

            if valid_loss_rolling_average_prev < valid_loss_rolling_average:
                logger.log('Stopping DynamicsEnsemble Training since valid_loss_rolling_average decreased')
                break
            valid_loss_rolling_average_prev = valid_loss_rolling_average

    def predict(self, obs, act):
        """
        Predict the batch of next observations given the batch of current observations and actions
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :return: pred_obs_next: predicted batch of next observations (n_samples, ndim_obs)
        """
        assert obs.shape[0] == act.shape[0]
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims

        obs_original = obs

        if self.normalize_input:
            obs, act = self._normalize_data(obs, act)
            delta = np.array(self.f_delta_pred(obs, act))
            delta = denormalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
        else:
            delta = np.array(self.f_delta_pred(obs, act))

        pred_obs = obs_original + delta
        return pred_obs

    def compute_normalization(self, obs, act, obs_next):
        """
        Computes the mean and std of the data and saves it in a instance variable
        -> the computed values are used to normalize the data at train and test time
        :param obs: observations - numpy array of shape (n_samples, ndim_obs)
        :param act: actions - numpy array of shape (n_samples, ndim_act)
        :param obs_next: observations after takeing action - numpy array of shape (n_samples, ndim_obs)
        """

        assert obs.shape[0] == obs_next.shape[0] == act.shape[0]
        delta = obs_next - obs
        assert delta.ndim == 2 and delta.shape[0] == obs_next.shape[0]

        # store means and std in dict
        self.normalization = OrderedDict()
        self.normalization['obs'] = (np.mean(obs, axis=0), np.std(obs, axis=0))
        self.normalization['delta'] = (np.mean(delta, axis=0), np.std(delta, axis=0))
        self.normalization['act'] = (np.mean(act, axis=0), np.std(act, axis=0))

    def _data_input_fn(self, obs, act, delta, batch_size=500, buffer_size=5000):
        """ Takes in train data an creates an a symbolic nex_batch operator as well as an iterator object """

        assert obs.ndim == act.ndim == delta.ndim, "inputs must have 2 dims"
        assert obs.shape[0] == act.shape[0] == delta.shape[0], "inputs must have same length along axis 0"
        assert obs.shape[1] == delta.shape[1], "obs and obs_next must have same length along axis 1 "

        self.obs_dataset_ph = tf.placeholder(tf.float32, (None, obs.shape[1]))
        self.act_dataset_ph = tf.placeholder(tf.float32, (None, act.shape[1]))
        self.delta_dataset_ph = tf.placeholder(tf.float32, (None, delta.shape[1]))

        dataset = tf.data.Dataset.from_tensor_slices(
            (self.obs_dataset_ph, self.act_dataset_ph, self.delta_dataset_ph)
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        return next_batch, iterator

    def _normalize_data(self, obs, act, obs_next=None):
        obs_normalized = normalize(obs, self.normalization['obs'][0], self.normalization['obs'][1])
        actions_normalized = normalize(act, self.normalization['act'][0], self.normalization['act'][1])

        if obs_next is not None:
            delta = obs_next - obs
            deltas_normalized = normalize(delta, self.normalization['delta'][0], self.normalization['delta'][1])
            return obs_normalized, actions_normalized, deltas_normalized
        else:
            return obs_normalized, actions_normalized

    def initialize_unitialized_variables(self, sess):
        uninit_variables = []
        for var in tf.global_variables():
            # note - this is hacky, may be better way to do this in newer TF.
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_variables.append(var)

        sess.run(tf.variables_initializer(uninit_variables))

    def reinit_model(self):
        sess = tf.get_default_session()
        if '_reinit_model_op' not in dir(self):
            self._reinit_model_op = [tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                    scope=self.name + '/dynamics_model'))]
        sess.run(self._reinit_model_op)

    def __getstate__(self):
        # state = LayersPowered.__getstate__(self)
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        state['normalization'] = self.normalization
        state['networks'] = [nn.__getstate__() for nn in self._networks]
        return state

    def __setstate__(self, state):
        # LayersPowered.__setstate__(self, state)
        Serializable.__setstate__(self, state['init_args'])
        self.normalization = state['normalization']
        for i in range(len(self._networks)):
            self._networks[i].__setstate__(state['networks'][i])


