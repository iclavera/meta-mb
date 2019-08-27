from meta_mb.utils.serializable import Serializable
import numpy as np
from meta_mb.logger import logger


class iLQRController(Serializable):
    def __init__(
            self,
            env,
            planner,
            num_rollouts,
            initializer_str,
            horizon,
    ):
        Serializable.quick_init(self, locals())
        self.initializer_str = initializer_str
        self.horizon = horizon
        self.num_envs = num_rollouts

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        planner.reset_u_array(self._init_u_array())
        self.planner = planner

    @property
    def vectorized(self):
        return True

    def get_sinusoid_actions(self, action_space, t):
        actions = np.array([])
        for i in range(action_space):
            actions = np.append(actions, 0.5 * np.sin(i * t))
        return actions

    def get_actions(self, obs, verbose=True):
        return self.planner.get_actions(obs, verbose=verbose)

    def _init_u_array(self):
        if self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.act_low, high=self.act_high,
                                             size=(self.horizon, self.num_envs, self.action_space_dims))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.num_envs, self.action_space_dims))
            init_u_array = np.clip(init_u_array, a_min=self.act_low, a_max=self.act_high)
        else:
            raise NotImplementedError
        return init_u_array

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        """
        This funciton is called by sampler at the start of each trainer iteration.
        :param dones:
        :return:
        """
        pass

    def warm_reset(self, u_array):
        logger.log('planner resets with collected samples...')
        if u_array is None or np.sum(np.abs(u_array) >= np.mean(self.act_high)) > 0.8 * (self.horizon*self.action_space_dims):
            u_array = self._init_u_array()
        else:
            u_array = u_array[:self.horizon, :, :]
        self.planner.reset_u_array(u_array[:self.horizon, :, :])

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
