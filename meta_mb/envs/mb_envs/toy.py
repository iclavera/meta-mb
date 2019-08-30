from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from collections import OrderedDict
import tensorflow as tf
import numpy as np


class Box(object):
    def __init__(self, low=None, high=None, shape=None):
        if shape is None:
            assert low.shape == high.shape
            shape = low.shape
        else:
            assert np.isscalar(low) and np.isscalar(high)
            low = low * np.ones(shape)
            high = high * np.ones(shape)
        self.shape = tuple(shape)
        self.dtype = np.float32
        self.low = low.astype(np.float32)
        self.high = high.astype(np.float32)

    def sample(self):
        return np.random.uniform(self.low, self.high, size=self.shape)


class FirstOrderEnv(object):

    def __init__(self):
        self.obs_dim = 2
        self.act_dim = 2
        self.init_pos = np.array([0, 0])
        self.goal_pos = np.array([10, 5])
        self.pos = None

        self.action_space = Box(low=-np.ones((2,)), high=np.ones((2,)))
        self.observation_space = Box(low=-np.inf*np.ones((2,)), high=np.inf*np.ones((2,)))

    def step(self, action):
        action = np.clip(action, self.action_space.low,
                         self.action_space.high)
        obs = self._get_obs()

        # do simulation
        self.pos = self.pos + action
        next_obs = self._get_obs()

        reward = self.reward(obs, action, next_obs)
        done = False
        return next_obs, reward, done, {}

    def tf_step(self, obs, act):
        # CALLER IS RESPONSIBLE FOR CLIPPING
        return obs + act

    def reset(self):
        self.set_state(self.init_pos + np.random.normal(loc=np.zeros_like(self.init_pos), scale=0.1))
        return self._get_obs()

    def _get_obs(self):
        return self.pos

    def set_state(self, pos):
        self.pos = pos

    def reward(self, obs, acts, next_obs):
        acts = np.clip(acts, self.action_space.low, self.action_space.high)
        if next_obs is not None:
            assert obs.shape == next_obs.shape
        if obs.ndim == 2:
            assert obs.shape[0] == acts.shape[0]
        else:
            assert obs.ndim == 1
        reward_run = -np.sum(np.square(self.goal_pos - obs), axis=-1)
        reward_ctrl = -0.1 * np.sum(np.square(acts), axis=-1)
        reward = reward_run + reward_ctrl
        return reward

    def tf_reward(self, obs, acts, next_obs):
        reward_run = -tf.reduce_sum(tf.square(self.goal_pos - obs), axis=-1)
        reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=-1)
        reward = reward_run + reward_ctrl
        return reward

    def get_goal_x_array(self, x_array):
        """

        :param x_array: (path_length, num_envs, obs_dim)
        :return:
        """
        path_length = x_array.shape[0]
        lin_interp = np.linspace(start=x_array[0, :, :], stop=self.goal_pos[None], num=path_length, endpoint=True)
        goal_x_array = np.minimum(x_array, self.goal_pos[None][None])
        goal_x_array = np.maximum(lin_interp, goal_x_array)
        assert goal_x_array.shape == x_array.shape

        return goal_x_array

    def reset_from_obs(self, obs):
        self.pos = obs
        return self._get_obs()

    def deriv_reward_obs(self, obs, acts):
        assert obs.ndim == acts.ndim
        return -2 * (obs - self.goal_pos[None])

    def deriv_reward_act(self, obs, acts):
        return -0.2 * acts

    def hessian_l_xx(self, obs, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, obs_dim, obs_dim)
        """
        hess = -2 * np.eye((self.obs_dim))
        hess = np.stack([hess for _ in range(obs.shape[0])], axis=0)
        return -hess

    def hessian_l_uu(self, obs, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, act_dim, act_dim)
        """
        hess = -0.2 * np.eye((self.obs_dim))
        hess = np.stack([hess for _ in range(obs.shape[0])], axis=0)
        return -hess

    def hessian_l_ux(self, obs, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, obs_dim, act_dim)
        """
        hess = np.zeros((obs.shape[0], self.act_dim, self.obs_dim))
        return -hess

    def dl_dict(self, inputs_dict):
        # FOR NEGATIVE RETURNS
        obs, acts = inputs_dict['obs'], inputs_dict['act']
        return OrderedDict(l_x=-self.deriv_reward_obs(obs, acts),
                           l_u=-self.deriv_reward_act(obs, acts),
                           l_xx=self.hessian_l_xx(obs, acts),
                           l_uu=self.hessian_l_uu(obs, acts),
                           l_ux=self.hessian_l_ux(obs, acts),)

    def log_diagnostics(self, paths, prefix):
        pass


if __name__ == "__main__":
    env = FirstOrderEnv()
    env.reset()

    fail_ctr = 0
    for _ in range(10000):
        x = np.random.random(size=(env.obs_dim,))
        new_x = env.reset_from_obs(x)
        if not np.allclose(x, new_x):
            fail_ctr += 1
            print(x, new_x)

    print(fail_ctr/100, ' percentage of failure')

    u_array = np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=(4, env.act_dim))
    obs = env.reset()

    for _ in range(10):
        x_array = [env.reset_from_obs(obs)]
        reward_array = []
        for i in range(4):
            x, reward, _, _ = env.step(u_array[i])
            x_array.append(x)
            reward_array.append(reward)

        print(f'x_array[0], reward_array = {x_array[0]}, {reward_array}')
        print(f'sum of rewards = {np.sum(reward_array)}')