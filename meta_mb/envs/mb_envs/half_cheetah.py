from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import MetaEnv


class HalfCheetahEnv(MetaEnv, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, frame_skip=5):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/half_cheetah.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]

    def step(self, action):
        start_ob = self._get_obs()
        reward_run = start_ob[8]

        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()

        reward = reward_run + reward_ctrl
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    # def reset_model(self):  # FIXME: hack for bptt
    #     self.set_state(self.init_qpos, self.init_qvel)
    #     return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reward(self, obs, acts, next_obs):
        if next_obs is not None:
            assert obs.shape == next_obs.shape
        if obs.ndim == 2:
            assert obs.shape[0] == acts.shape[0]
            reward_ctrl = -0.1 * np.sum(np.square(acts), axis=1)
            reward_run = obs[:, 8]
            reward = reward_run + reward_ctrl
        elif obs.ndim == 1:
            reward_ctrl = -0.1 * np.sum(np.square(acts))
            reward_run = obs[8]
            reward = reward_run + reward_ctrl  # scalar
        else:
            raise ValueError
        return reward

    def get_goal_x_array(self, x_array):
        """

        :param x_array: (path_length, num_envs, obs_dim)
        :return:
        """
        goal_x_array = x_array.copy()
        if goal_x_array.ndim == 3:
            goal_x_array[:, :, 8] = np.minimum(goal_x_array[:, :, 8], 4)
        else:
            goal_x_array[:, 8] = np.minimum(goal_x_array[:, 8], 4)
        return goal_x_array

    def tf_reward(self, obs, acts, next_obs):
        acts = tf.clip_by_value(acts, self.action_space.low, self.action_space.high)
        if obs.get_shape().ndims == 1:
            reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=0)
            reward_run = obs[8]
        elif obs.get_shape().ndims == 2:
            reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
            reward_run = obs[:, 8]  # changed from next_obs to obs
        else:
            raise NotImplementedError
        reward = reward_run + reward_ctrl
        return reward

    def tf_deriv_reward_obs(self, obs, acts, batch_size):
        deriv = np.zeros((batch_size, self.obs_dim))
        deriv[:, 8] = 1
        return tf.constant(deriv, dtype=tf.float32)

    def tf_deriv_reward_act(self, obs, acts, batch_size):
        return -0.2*acts

    def tf_hessian_l_xx(self, obs, acts, batch_size):
        hess = tf.zeros((batch_size, self.obs_dim, self.obs_dim))
        return -hess

    def tf_hessian_l_uu(self, obs, acts, batch_size):
        hess = np.zeros((batch_size, self.act_dim, self.act_dim))
        for i in range(self.act_dim):
            hess[:, i, i] = -0.2
        return tf.constant(-hess)

    def tf_hessian_l_ux(self, obs, acts, batch_size):
        hess = tf.zeros((batch_size, self.act_dim, self.obs_dim))
        return -hess

    def reset_from_obs(self, obs):
        nq, nv = self.model.nq, self.model.nv
        self.sim.reset()
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.1, high=.1, size=nq)
        qpos[1:] = obs[:nq-1]
        qvel = obs[nq-1:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def deriv_reward_obs(self, obs, acts):
        assert obs.ndim == acts.ndim
        if obs.ndim == 1:
            deriv = np.zeros_like(obs)
            deriv[8] = 1
        elif obs.ndim == 2:
            deriv = np.zeros_like(obs)
            deriv[:, 8] = 1
        else:
            raise NotImplementedError
        return deriv

    def deriv_reward_act(self, obs, acts):
        assert obs.ndim == acts.ndim
        return -0.2*acts #(-0.2 * acts).copy()

    def hessian_l_xx(self, obs, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, obs_dim, obs_dim)
        """
        hess = np.zeros((obs.shape[0], self.obs_dim, self.obs_dim))
        return -hess

    def hessian_l_uu(self, obs, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, act_dim, act_dim)
        """
        hess = np.zeros((obs.shape[0], self.act_dim, self.act_dim))
        for i in range(self.act_dim):
            hess[:, i, i] = -0.2
        return -hess

    def hessian_l_ux(self, obs, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, obs_dim, act_dim)
        """
        hess = np.zeros((obs.shape[0], self.act_dim, self.obs_dim))
        return -hess

    def tf_dl_dict(self, obs, acts, next_obs, batch_size):
        return OrderedDict(
            l_x=-self.tf_deriv_reward_obs(obs, acts, batch_size),
            l_u=-self.tf_deriv_reward_act(obs, acts, batch_size),
            l_xx=self.tf_hessian_l_xx(obs, acts, batch_size),
            l_uu=self.tf_hessian_l_uu(obs, acts, batch_size),
            l_ux=self.tf_hessian_l_ux(obs, acts, batch_size),
        )

    def dl_dict(self, inputs_dict):
        # FOR NEGATIVE RETURNS
        obs, acts = inputs_dict['obs'], inputs_dict['act']
        return OrderedDict(l_x=-self.deriv_reward_obs(obs, acts),
                           l_u=-self.deriv_reward_act(obs, acts),
                           l_xx=self.hessian_l_xx(obs, acts),
                           l_uu=self.hessian_l_uu(obs, acts),
                           l_ux=self.hessian_l_ux(obs, acts),)


if __name__ == "__main__":
    env = HalfCheetahEnv()
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

    # print(env.sim.derivative().shape)
    # print(env.sim.data.qpos)

        # for _ in range(1000):
        #     _ = env.render()
        #     ob, rew, done, info = env.step(env.action_space.sample())  # take a random action

    # for _ in range(10):
    #     ob, reward, done, info = env.step(np.zeros(env.action_space.shape))
    #     print(ob)
    #     print(reward)