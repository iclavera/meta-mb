import numpy as np
from meta_mb.utils.filters import MeanStdFilter
from meta_mb.utils import Serializable
from meta_mb.policies.np_base import NpPolicy
from collections import OrderedDict

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
        self.policy_params = OrderedDict(W=np.zeros((action_dim, obs_dim), dtype=np.float64))
        # TODO: Create my own filter this will be painfully slow!!!
        self.obs_filter = MeanStdFilter(shape=(obs_dim,))

    def get_actions(self, observations, update_filter=True):
        obs = self.obs_filter(observations, update=update_filter)
        return np.dot(self.policy_params["W"], obs.T).T, {}

    def get_action(self, observation, update_filter=False):
        return self.get_actions(np.expand_dims(observation, axis=0), update_filter=update_filter)[0], {}

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
            obs = self.obs_filter(obs, update=update_filter)
            obs = np.reshape(obs, (self._num_deltas, -1, self.obs_dim))
            actions = np.matmul(self.policy_params_batch["W"], obs.transpose((0, 2, 1))).transpose((0, 2, 1))
            assert actions.shape == (self._num_deltas, observations.shape[1], self.action_dim)
        elif observations.ndim == 2:
            obs = self.obs_filter(observations, update=update_filter)
            obs = np.expand_dims(obs, axis=1)
            actions = np.matmul(self.policy_params_batch["W"], obs.transpose((0, 2, 1))).transpose((0, 2, 1))
            actions = actions[:, 0, :]
        else:
            raise NotImplementedError
        return actions, {}

    def get_stats_filter(self):
        return self.obs_filter.get_stats()
