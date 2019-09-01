import numpy as np
from meta_mb.utils.filters import MeanStdFilter, TfMeanStdFilter
from meta_mb.policies.distributions.diagonal_gaussian import DiagonalGaussian
from meta_mb.policies.np_base import NpPolicy
from collections import OrderedDict
import tensorflow as tf
from pdb import set_trace as st

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class LinearPolicy(NpPolicy):
    """
    Linear policy class that computes action as <W, ob>.
    """

    def __init__(self,
                 obs_dim,
                 action_dim,
                 name='np_policy',
                 **kwargs):
        NpPolicy.__init__(self, obs_dim, action_dim, name, **kwargs)
        self.policy_params = OrderedDict(W=np.zeros((action_dim, obs_dim), dtype=np.float64),
                                         b=np.zeros((action_dim,), dtype=np.float64))
        self.obs_filters = [MeanStdFilter(shape=(obs_dim,))]
        self.tf_obs_filters = [TfMeanStdFilter(shape=(obs_dim,))]
        # self._dist = DiagonalGaussian(self.action_dim, squashed=True)


    def get_actions(self, observations, update_filter=True):
        observations = np.array(observations)
        assert observations.ndim == 2
        obs = self.obs_filters[0](observations, update=update_filter)
        actions = np.dot(self.policy_params["W"], obs.T).T + self.policy_params["b"]
        return actions, {}

    def get_action(self, observation, update_filter=False):
        actions, _ = self.get_actions(np.expand_dims(observation, axis=0), update_filter=update_filter)
        return actions[0], {}


    def get_actions_batch(self, observations, update_filter=True):
        """
        The observations must be of shape num_deltas x batch_size x obs_dim
        :param observations:
        :param update_filter:
        :return:
        """
        # TODO: make sure the reshaping works
        assert observations.shape[0] == self._num_deltas and observations.shape[-1] == self.obs_dim
        if observations.ndim == 3:
            obs = np.reshape(observations, (-1, self.obs_dim))
            obs = self.obs_filters[0](obs, update=update_filter)
            obs = np.reshape(obs, (self._num_deltas, -1, self.obs_dim))
            actions = np.matmul(self.policy_params_batch["W"], obs.transpose((0, 2, 1))).transpose((0, 2, 1))\
                      + np.expand_dims(self.policy_params_batch["b"], axis=1)
            assert actions.shape == (self._num_deltas, observations.shape[1], self.action_dim)
        elif observations.ndim == 2:
            obs = self.obs_filters[0](observations, update=update_filter)
            obs = np.expand_dims(obs, axis=1)
            actions = np.matmul(self.policy_params_batch["W"], obs.transpose((0, 2, 1))).transpose((0, 2, 1)) \
                    + np.expand_dims(self.policy_params_batch["b"], axis=1)
            actions = actions[:, 0, :]
        else:
            raise NotImplementedError
        return actions, {}

class TfLinearPolicy(NpPolicy):
    """
    Linear policy class that computes action as <W, ob>.
    """

    def __init__(self,
                 obs_dim,
                 action_dim,
                 name='np_policy',
                 # squashed=True,
                 **kwargs):
        NpPolicy.__init__(self, obs_dim, action_dim, name, **kwargs)
        self.policy_params = OrderedDict(W = tf.get_variable('W',
                                                             shape = (action_dim, obs_dim),
                                                             initializer=tf.zeros_initializer()),
                                        b = tf.get_variable('b',
                                                            shape = (action_dim),
                                                            initializer=tf.zeros_initializer()))

        self.tf_obs_filters = [TfMeanStdFilter(shape=(obs_dim,))]
        self.squashed = True
        self.build_graph()

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        self.obs_ph = tf.placeholder(tf.float32, shape = [None, self.obs_dim])
        self.action_var = tf.transpose(tf.matmul(tf.cast(self.policy_params['W'], self.obs_ph.dtype), tf.transpose(self.obs_ph))) + tf.cast(self.policy_params['b'], self.obs_ph.dtype)
        current_scope = self.name


    def tf_get_actions(self, observations, update_filter=True, params = None):
        assert len(observations.shape) == 2
        st()
        obs = self.tf_obs_filters[0](observations, update=update_filter)
        if params == None:
            actions = tf.transpose(tf.matmul(self.policy_params["W"], tf.transpose(obs))) + self.policy_params["b"]
        else:
            actions = tf.transpose(tf.matmul(tf.cast(params["W"], obs.dtype), tf.transpose(obs))) + tf.cast(params["b"], obs.dtype)
        actions = tf.clip_by_value(actions, LOG_SIG_MIN, LOG_SIG_MAX)
        return actions, {}

    def tf_get_action(self, observation, update_filter=False, params = None):
        actions, _ = self.tf_get_actions(observation, update_filter=update_filter, params = params)
        return actions, {}

    def get_actions(self, observations, update_filter=True):
        observations = np.array(observations)
        assert observations.ndim == 2
        sess = tf.get_default_session()
        actions = sess.run([self.action_var],feed_dict={self.obs_ph: observations})
        actions = np.tanh(actions)
        return actions, {}

    def get_action(self, observation, update_filter=False):
        if observation.ndim == 1:
            actions, _ = self.get_actions(np.expand_dims(observation, axis=0), update_filter=update_filter)
        else:
            actions, _ = self.get_actions(observation, update_filter=update_filter)

        return actions[0], {}

    # def get_actions_batch(self, observations, update_filter=True):
    #     """
    #     The observations must be of shape num_deltas x batch_size x obs_dim
    #     :param observations:
    #     :param update_filter:
    #     :return:
    #     """
    #     # TODO: make sure the reshaping works
    #     assert observations.shape[0] == self._num_deltas and observations.shape[-1] == self.obs_dim
    #     if observations.ndim == 3:
    #         obs = np.reshape(observations, (-1, self.obs_dim))
    #         obs = self.obs_filters[0](obs, update=update_filter)
    #         obs = np.reshape(obs, (self._num_deltas, -1, self.obs_dim))
    #         actions = np.matmul(self.policy_params_batch["W"], obs.transpose((0, 2, 1))).transpose((0, 2, 1))\
    #                   + np.expand_dims(self.policy_params_batch["b"], axis=1)
    #         assert actions.shape == (self._num_deltas, observations.shape[1], self.action_dim)
    #     elif observations.ndim == 2:
    #         obs = self.obs_filters[0](observations, update=update_filter)
    #         obs = np.expand_dims(obs, axis=1)
    #         actions = np.matmul(self.policy_params_batch["W"], obs.transpose((0, 2, 1))).transpose((0, 2, 1)) \
    #                 + np.expand_dims(self.policy_params_batch["b"], axis=1)
    #         actions = actions[:, 0, :]
    #     else:
    #         raise NotImplementedError
    #     return actions, {}
