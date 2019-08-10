from meta_mb.utils.serializable import Serializable
import numpy as np
from gym.spaces import Box

class ObsStackEnv(Serializable):
    def __init__(self, env, time_steps=3):
        Serializable.quick_init(self, locals())

        self._wrapped_env = env
        self._time_steps = time_steps

        self._orig_obs_dim = self._wrapped_env.observation_space.shape[0]

    def step(self, action):
        obs, reward, done, info = self._wrapped_env.step(action)

        self._obs[self._orig_obs_dim:] = self._obs[:-self._orig_obs_dim]
        self._obs[:self._orig_obs_dim] = obs

        return self._obs.copy(), reward, done, info


    def reset(self):
        obs = self._wrapped_env.reset()
        self._obs = np.zeros(self._orig_obs_dim * self._time_steps)

        self._obs[:self._orig_obs_dim] = obs

        return self._obs.copy()

    @property
    def observation_space(self):
        assert len(self._wrapped_env.observation_space.shape) == 1
        return Box(low=-1e-6, high=1e6, shape=(self._orig_obs_dim * self._time_steps, ))

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        """
        orig_attr = self._wrapped_env.__getattribute__(attr)
        # if hasattr(self._wrapped_env, '_wrapped_env'):
        #     orig_attr = self._wrapped_env.__getattr__(attr)
        # else:
        #     orig_attr = self._wrapped_env.__getattribute__(attr)
        #
        # if callable(orig_attr):
        #     def hooked(*args, **kwargs):
        #         result = orig_attr(*args, **kwargs)
        #         return result
        #
        #     return hooked
        # else:
        #     return orig_attr
        return orig_attr
