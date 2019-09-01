from meta_mb.utils.serializable import Serializable
import numpy as np
from meta_mb.logger import logger


class iLQRController(Serializable):
    def __init__(
            self,
            env,
            planner,
            num_rollouts,
            horizon,
    ):
        Serializable.quick_init(self, locals())
        self.horizon = horizon
        self.num_envs = num_rollouts

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        # planner.reset_u_array(self._init_u_array())
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

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        """
        This funciton is called by sampler at the start of each trainer iteration.
        :param dones:
        :return:
        """
        pass
        # self.planner.u_array_val = None

    def warm_reset(self, u_array):
        pass
        # logger.log('planner resets with collected samples...')
        if u_array is None or np.sum(np.abs(u_array) >= np.mean(self.act_high)) > 0.8 * (self.horizon*self.action_space_dims):
            u_array = None
        else:
            u_array = u_array[:self.horizon, :, :]
        self.planner.warm_reset(u_array=u_array)

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
