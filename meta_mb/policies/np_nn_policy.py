import numpy as np
from meta_mb.utils.filters import MeanStdFilter
from meta_mb.utils import Serializable
from meta_mb.policies.np_base import NpPolicy
from collections import OrderedDict

class NNPolicy(NpPolicy):
    """
    Linear policy class that computes action as <W, ob>.
    """

    _activations = {
        'tanh': np.tanh,
        None: lambda x: x,
        'relu': lambda x: np.maximum(x, 0)
    }

    def __init__(self,
                 obs_dim,
                 action_dim,
                 name='np_policy',
                 hidden_sizes=(64, 64),
                 hidden_nonlinearity='tanh',
                 output_nonlinearity=None,
                 **kwargs):
        NpPolicy.__init__(self, obs_dim, action_dim, name, **kwargs)
        self.obs_filter = MeanStdFilter(shape=(obs_dim,))
        self.hidden_nonlinearity = self._activations[hidden_nonlinearity]
        self.output_nonlinearity = self._activations[output_nonlinearity]
        self.hidden_sizes = hidden_sizes
        self.policy_params = OrderedDict()
        prev_size = action_dim
        for i, hidden_size in enumerate(hidden_sizes):
            W = np.zeros((prev_size, hidden_size), dtype=np.float64)
            b = np.zeros((hidden_size,))

            self.policy_params['W_%d' % i] = W
            self.policy_params['b_%d' % i] = b
            prev_size = hidden_size

        W = np.zeros((prev_size, obs_dim), dtype=np.float64)
        b = np.zeros((obs_dim,))
        self.policy_params['W_out'] = W
        self.policy_params['b_out'] = b

    def get_actions(self, observations, update_filter=True):
        x = self.obs_filter(observations, update=update_filter)
        for i, hidden_size in enumerate(self.hidden_sizes):
            x = np.dot(self.policy_params['W_%d' % i], x.T).T + self.policy_params['b_%d' % i]
            x = self.hidden_nonlinearity(x)
        actions = self.output_nonlinearity(np.dot(self.policy_params['W_out'], x.T).T + self.policy_params['b_out'])
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
        assert observations.shape[0] == self._num_deltas and observations.shape[-1] == self.obs_dim
        if observations.ndim == 3:
            obs = np.reshape(observations, (-1, self.obs_dim))
            obs = self.obs_filter(obs, update=update_filter)
            x = np.reshape(obs, (self._num_deltas, -1, self.obs_dim))
        elif observations.ndim == 2:
            obs = self.obs_filter(observations, update=update_filter)
            x = np.expand_dims(obs, axis=1)
        else:
            raise NotImplementedError

        for i, hidden_size in enumerate(self.hidden_sizes):
            x = self.hidden_nonlinearity(np.matmul(self.policy_params_batch['W_%d' % i],
                                         x.transpose((0, 2, 1))).transpose((0, 2, 1))
                                         + np.expand_dims(self.policy_params_batch['b_%d' % i], axis=1))

        actions = self.output_nonlinearity(np.matmul(self.policy_params_batch['W_out'],
                                                     x.transpose((0, 2, 1))).transpose((0, 2, 1))
                                                     + np.expand_dims(self.policy_params_batch['b_out'], axis=1))

        if observations.ndim == 2:
            actions = actions[:, 0, :]
        return actions, {}

    def get_stats_filter(self):
        return self.obs_filter.get_params()
