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

    def step(self, action):
        start_ob = self._get_obs()
        reward_run = start_ob[8]

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)
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

    def reset_model(self):  # FIXME: hack for bptt
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        if next_obs is not None:
            assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        reward_ctrl = -0.1 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, 8]
        reward = reward_run + reward_ctrl
        return reward

    def tf_reward(self, obs, acts, next_obs):
        reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
        reward_run = obs[:, 8]  # changed from next_obs to obs
        reward = reward_run + reward_ctrl
        return reward

    def reset_from_obs(self, obs):
        nq, nv = self.model.nq, self.model.nv
        self.sim.reset()
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.1, high=.1, size=nq)
        qpos[1:] = obs[:nq-1]
        qvel = obs[nq-1:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def deriv_reward_obses(self, obses, acts):
        deriv = np.zeros_like(obses)
        deriv[:, 8] = 1
        return deriv

    def deriv_reward_acts(self, obses, acts):
        return -0.2 * acts[:, :]

    def hessian_l_xx(self, obses, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, obs_dim, obs_dim)
        """
        hess = np.zeros((obses.shape[0], self.obs_dim, self.obs_dim))
        return -hess

    def hessian_l_uu(self, obses, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, act_dim, act_dim)
        """
        hess = np.zeros((obses.shape[0], self.act_dim, self.act_dim))
        for i in range(self.act_dim):
            hess[:, i, i] = -0.2
        return -hess

    def hessian_l_ux(self, obses, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, obs_dim, act_dim)
        """
        hess = np.zeros((obses.shape[0], self.act_dim, self.obs_dim))
        return -hess

    def dl_dict(self, inputs_dict):
        # FOR NEGATIVE RETURNS
        obses, acts = inputs_dict['obs'], inputs_dict['act']
        return OrderedDict(l_x=-self.deriv_reward_obses(obses, acts),
                           l_u=-self.deriv_reward_acts(obses, acts),
                           l_xx=self.hessian_l_xx(obses, acts),
                           l_uu=self.hessian_l_uu(obses, acts),
                           l_ux=self.hessian_l_ux(obses, acts),)


if __name__ == "__main__":
    env = HalfCheetahEnv()
    env.reset()

    # print(env.sim.derivative().shape)
    # print(env.sim.data.qpos)

    for _ in range(1000):
        _ = env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action

    # for _ in range(10):
    #     ob, reward, done, info = env.step(np.zeros(env.action_space.shape))
    #     print(ob)
    #     print(reward)