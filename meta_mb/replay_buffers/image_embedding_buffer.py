import numpy as np
import tensorflow as tf
from collections import OrderedDict

from meta_mb.utils.serializable import Serializable

from meta_mb.logger import logger
from meta_mb.dynamics.utils import normalize, denormalize, train_test_split

class ImageEmbeddingBuffer(Serializable):
    def __init__(self, batch_size, env, encoder, return_image, latent_dim, num_stack, num_models, valid_split_ratio, buffer_size=500,
                 normalize_input=True):
        self._dataset = None
        self._embedding_dataset = None
        self._train_idx = None
        self._buffer_size = buffer_size

        self._batch_size = batch_size

        # determine dimensionality of state and action space
        self.obs_space_dims = env.observation_space.shape
        self.action_space_dims = env.action_space.shape[0]
        self.encoder = encoder
        self.return_image = return_image
        self._latent_dim = latent_dim
        self._num_stack = num_stack

        self.timesteps_counter = 0

        self._num_models = num_models
        self._valid_split_ratio=valid_split_ratio

        self._normalize_input = normalize_input
        self._normalization = None

        self._create_stats_vars()

    def update_buffer(self, obs, act, obs_next, reward, check_init=True):
        """
        :param obs: shape N x T x img_size
        :param act: shape N x T x ac_dim
        :param obs_next: shape N x T x img_size
        """

        assert obs.ndim == 5 and obs.shape[2:] == self.obs_space_dims
        assert obs_next.ndim == 5 and obs_next.shape[2:] == self.obs_space_dims
        assert act.ndim == 3 and act.shape[2] == self.action_space_dims
        assert reward.ndim == 2

        self.timesteps_counter += obs.shape[0]
        obs_seq = np.concatenate([obs, obs_next[:, -1:, :]], axis=1)
        # If case should be entered exactly once
        if check_init and self._dataset is None:
            self._dataset = dict(obs=obs_seq, act=act, reward=reward)
            self.update_train_idx(self._valid_split_ratio)

        else:
            n_new_samples = len(obs)
            n_max = self._buffer_size - n_new_samples

            self._dataset['obs'] = np.concatenate([self._dataset['obs'][-n_max:], obs_seq])
            self._dataset['act'] = np.concatenate([self._dataset['act'][-n_max:], act])
            self._dataset['reward'] = np.concatenate([self._dataset['reward'][-n_max:], reward])

            self.update_train_idx(self._valid_split_ratio, n_new_samples=n_new_samples)

        logger.log('Model has dataset with size {}'.format(len(self._dataset['obs'])))

    def update_embedding_buffer(self, check_init=True):
        if check_init and self._embedding_dataset is None:
            self._embedding_dataset = {}
            assert self._normalization is None

        self._embedding_dataset['obs'] = self.encoder.predict(self._dataset['obs'].reshape(-1, *self.obs_space_dims))
        self._embedding_dataset['obs'] = np.reshape(self._embedding_dataset['obs'],
                                                    self._dataset['obs'].shape[:2] + (self._latent_dim,))
        self._embedding_dataset['act'] = self._dataset['act']
        self._embedding_dataset['delta'] = self._embedding_dataset['obs'][:, 1:] - self._embedding_dataset['obs'][:,
                                                                                   :-1]
        self._embedding_dataset['reward'] = self._dataset['reward']

        if self._normalize_input:
            self.compute_normalization(self._embedding_dataset['obs'],
                                       self._embedding_dataset['act'],
                                       self._embedding_dataset['delta'],
                                       self._embedding_dataset['reward'])
            self._embedding_dataset['obs'], self._embedding_dataset['act'], self._embedding_dataset['delta'],\
            self._embedding_dataset['reward'] = \
                self._normalize_data(self._embedding_dataset['obs'], self._embedding_dataset['act'],
                                     self._embedding_dataset['delta'], self._embedding_dataset['reward'])

    def update_train_idx(self, valid_split_ratio, n_new_samples=0):
        if self._train_idx is None:
            self._train_idx = []
            dataset_size = len(self._dataset['obs'])
            train_size = int(dataset_size * (1 - valid_split_ratio))
            for _ in range(self._num_models):
                indices = np.random.choice(dataset_size, train_size, replace=False)
                train_idx = np.zeros(shape=dataset_size)
                train_idx[indices] = 1
                self._train_idx.append(train_idx)
        else:
            assert n_new_samples
            n_max = self._buffer_size - n_new_samples

            for i in range(self._num_models):
                old_train_idx = self._train_idx[i]

                train_size = int(self._dataset['obs'].shape[0] * (1 - valid_split_ratio)) - np.sum(
                    old_train_idx[-n_max:]).astype(int)
                indices = np.random.choice(n_new_samples, train_size, replace=False)
                train_idx = np.zeros(shape=n_new_samples)
                train_idx[indices] = 1
                self._train_idx[i] = np.concatenate([old_train_idx[-n_max:], train_idx])

    def generate_batch(self, test=False):
        if self.return_image:
            data_act, data_obs, data_delta = self._dataset['act'], self._dataset['obs'], \
                                             self._dataset['obs'][:, 1:]
        else:
            data_act, data_obs, data_delta = self._embedding_dataset['act'], self._embedding_dataset['obs'], \
                                            self._embedding_dataset['delta']
        ret_obs = []
        ret_delta = []
        ret_actions = []
        data_obs = np.concatenate([np.zeros((data_obs.shape[0], self._num_stack - 1, *data_obs.shape[2:])), data_obs],
                                  axis=1)
        obs_stack = np.stack([data_obs[:, offset: data_obs.shape[1] + offset - self._num_stack + 1]
                              for offset in range(self._num_stack)], axis=2)[:, :-1]
        assert len(self._train_idx[0]) == len(data_act) == len(data_obs), 'the three are %d, %d, and %d respectively' \
                                                                          % (len(self._train_idx[0]), len(data_act),
                                                                             len(data_obs))

        for i in range(self._num_models):
            mask = 1 - self._train_idx[i] if test else self._train_idx[i]
            select_idx = np.arange(data_act.shape[0])[mask.astype(bool)]
            obs = np.concatenate(obs_stack[select_idx])
            actions = np.concatenate(data_act[select_idx], axis=0)
            delta = np.concatenate(data_delta[select_idx], axis=0)

            assert obs.shape[0] == actions.shape[0] == delta.shape[0]

            batch_idx = np.random.choice(obs.shape[0], size=self._batch_size)

            ret_obs.append(obs[batch_idx])
            ret_actions.append(actions[batch_idx])
            ret_delta.append(delta[batch_idx])

        return np.concatenate(ret_obs), np.concatenate(ret_actions), np.concatenate(ret_delta)

    def generate_reward_batch(self, test=False):
        if self.return_image:
            data_act, data_obs, data_reward = self._dataset['act'], self._dataset['obs'], \
                                             self._dataset['reward']
        else:
            data_act, data_obs, data_reward = self._embedding_dataset['act'], self._embedding_dataset['obs'], \
                                              self._embedding_dataset['reward']
        ret_obs = []
        ret_nextobs = []
        ret_actions = []
        ret_reward = []
        data_nextobs = data_obs[:, 1:]
        data_obs = data_obs[:, :-1]


        assert len(self._train_idx[0]) == len(data_act) == len(data_obs) == len(data_nextobs) == len(data_reward), \
            'the three are %d, %d, and %d respectively'  % (len(self._train_idx[0]), len(data_act),
                                                                             len(data_obs))

        for i in range(self._num_models):
            mask = 1 - self._train_idx[i] if test else self._train_idx[i]
            select_idx = np.arange(data_act.shape[0])[mask.astype(bool)]
            obs = np.concatenate(data_obs[select_idx], axis=0)
            actions = np.concatenate(data_act[select_idx], axis=0)
            nextobs = np.concatenate(data_nextobs[select_idx], axis=0)
            reward = np.concatenate(data_reward[select_idx], axis=0)

            assert obs.shape[0] == actions.shape[0] == nextobs.shape[0] == reward.shape[0]

            if not test:
                batch_idx = np.random.choice(obs.shape[0], size=self._batch_size)

            ret_obs.append(obs[batch_idx])
            ret_actions.append(actions[batch_idx])
            ret_nextobs.append(nextobs[batch_idx])
            ret_reward.append(reward[batch_idx])

        return np.concatenate(ret_obs), np.concatenate(ret_actions), np.concatenate(ret_nextobs), \
               np.concatenate(ret_reward).reshape((-1, 1))

    def compute_normalization(self, obs, act, delta, reward):
        assert obs.shape[0] == act.shape[0] == delta.shape[0] == reward.shape[0]
        assert obs.shape[1] == act.shape[1] + 1 == delta.shape[1] + 1 == reward.shape[1] + 1
        obs = obs.reshape((-1, self._latent_dim))
        act = act.reshape((-1, self.action_space_dims))
        delta = delta.reshape((-1, self._latent_dim))
        reward = reward.reshape((-1))

        # store means and std in dict

        feed_dict = {}

        normalization = OrderedDict()
        normalization['obs'] = (np.mean(obs, axis=0), np.std(obs, axis=0))
        normalization['delta'] = (np.mean(delta, axis=0), np.std(delta, axis=0))
        normalization['act'] = (np.mean(act, axis=0), np.std(act, axis=0))
        normalization['reward'] = (np.mean(reward, axis=0), np.std(reward, axis=0))
        self._normalization = normalization
        feed_dict.update({self._mean_obs_ph: self._normalization['obs'][0],
                          self._std_obs_ph: self._normalization['obs'][1],
                          self._mean_act_ph: self._normalization['act'][0],
                          self._std_act_ph: self._normalization['act'][1],
                          self._mean_delta_ph: self._normalization['delta'][0],
                          self._std_delta_ph: self._normalization['delta'][1],
                          self._mean_reward_ph: self._normalization['reward'][0],
                          self._std_reward_ph: self._normalization['reward'][1],
                          }
                         )
        sess = tf.get_default_session()
        sess.run(self._assignations, feed_dict=feed_dict)

    def _create_stats_vars(self):
        self._assignations = []

        self._mean_obs_var = tf.get_variable('mean_obs', shape=(self._latent_dim,),
                                             dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
        self._std_obs_var = tf.get_variable('std_obs', shape=(self._latent_dim,),
                                            dtype=tf.float32, initializer=tf.ones_initializer, trainable=False)
        self._mean_act_var = tf.get_variable('mean_act', shape=(self.action_space_dims,),
                                             dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
        self._std_act_var = tf.get_variable('std_act', shape=(self.action_space_dims,),
                                            dtype=tf.float32, initializer=tf.ones_initializer, trainable=False)
        self._mean_delta_var = tf.get_variable('mean_delta', shape=(self._latent_dim,),
                                               dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
        self._std_delta_var = tf.get_variable('std_delta', shape=(self._latent_dim,),
                                              dtype=tf.float32, initializer=tf.ones_initializer, trainable=False)
        self._mean_reward_var = tf.get_variable('mean_reward', shape=(),
                                               dtype=tf.float32, initializer=tf.zeros_initializer, trainable=False)
        self._std_reward_var = tf.get_variable('std_reward', shape=(),
                                              dtype=tf.float32, initializer=tf.ones_initializer, trainable=False)

        self._mean_obs_ph = tf.placeholder(tf.float32, shape=(self._latent_dim,))
        self._std_obs_ph = tf.placeholder(tf.float32, shape=(self._latent_dim,))
        self._mean_act_ph = tf.placeholder(tf.float32, shape=(self.action_space_dims,))
        self._std_act_ph = tf.placeholder(tf.float32, shape=(self.action_space_dims,))
        self._mean_delta_ph = tf.placeholder(tf.float32, shape=(self._latent_dim,))
        self._std_delta_ph = tf.placeholder(tf.float32, shape=(self._latent_dim,))
        self._mean_reward_ph = tf.placeholder(tf.float32, shape=())
        self._std_reward_ph = tf.placeholder(tf.float32, shape=())

        self._assignations.extend([tf.assign(self._mean_obs_var, self._mean_obs_ph),
                                   tf.assign(self._std_obs_var, self._std_obs_ph),
                                   tf.assign(self._mean_act_var, self._mean_act_ph),
                                   tf.assign(self._std_act_var, self._std_act_ph),
                                   tf.assign(self._mean_delta_var, self._mean_delta_ph),
                                   tf.assign(self._std_delta_var, self._std_delta_ph),
                                   tf.assign(self._mean_reward_var, self._mean_reward_ph),
                                   tf.assign(self._std_reward_var, self._std_reward_ph),
                                   ])

    def _normalize_data(self, obs, act, delta, reward):
        assert self._normalization is not None

        norm_obs = normalize(obs, self._normalization['obs'][0], self._normalization['obs'][1])
        norm_act = normalize(act, self._normalization['act'][0], self._normalization['act'][1])
        norm_delta = normalize(delta, self._normalization['delta'][0], self._normalization['delta'][1])
        norm_reward = normalize(reward, self._normalization['reward'][0], self._normalization['reward'][1])

        return norm_obs, norm_act, norm_delta, norm_reward

    @property
    def size(self):
        if self._dataset is None:
            return 0
        else:
            return int(self._dataset['act'].shape[0] * (1 - self._valid_split_ratio)) * self._dataset['act'].shape[1]

    @property
    def val_size(self):
        if self._dataset is None:
            return 0
        else:
            return int(self._dataset['act'].shape[0] * self._valid_split_ratio) * self._dataset['act'].shape[1]

