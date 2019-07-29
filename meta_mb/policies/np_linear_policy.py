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
        self.output_nonlinearity = kwargs.get('output_nonlinearity', None)
        NpPolicy.__init__(self, obs_dim, action_dim, name, **kwargs)
        # self.policy_params = OrderedDict(W=np.zeros((action_dim, obs_dim), dtype=np.float64),
        #                                  b=np.zeros((action_dim,), dtype=np.float64))
        self.policy_params = OrderedDict(
            W=np.zeros((action_dim, obs_dim), dtype=np.float64),
            b=np.random.normal(loc=0, scale=0.5, size=(action_dim,))
        )
        self.obs_filters = [MeanStdFilter(shape=(obs_dim,))]

    def get_actions(self, observations, update_filter=True):
        observations = np.array(observations)
        assert observations.ndim == 2
        obs = self.obs_filters[0](observations, update=update_filter)
        actions = np.dot(self.policy_params["W"], obs.T).T + self.policy_params["b"]
        if self.output_nonlinearity is not None:
            actions = self.output_nonlinearity(actions)
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
        if self.output_nonlinearity is not None:
            actions = self.output_nonlinearity(actions)
        return actions, {}

    def perturb_W(self, i, j, eps):
        self.policy_params['W'][i, j] += eps

    def perturb_b(self, i, eps):
        self.policy_params['b'][i] += eps

    def add_to_params(self, delta_policy_params):
        assert all([k1 == k2 for k1, k2 in zip(self.get_params().keys(), delta_policy_params.keys())]), \
            "parameter keys must match with variable"

        for k, v in delta_policy_params.items():
            self.policy_params[k] += v
