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


class HalfCheetahEnvQuadReward(MetaEnv, mujoco_env.MujocoEnv, utils.EzPickle):

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

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reward(self, obs, acts, next_obs):
        # if next_obs is not None:
        assert obs.shape == obs.shape
        if obs.ndim == 2:  # (batch_size, act_dim)
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

    def reset_from_obs(self, obs):
        nq, nv = self.model.nq, self.model.nv
        self.sim.reset()
        # FIXME: noise here?
        qpos = self.init_qpos + \
             self.np_random.uniform(low=-.1, high=.1, size=nq)
        qpos[1:] = obs[:nq-1]
        qvel = obs[nq-1:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def tf_reward(self, obs, acts, next_obs):
        # FIXME: SHOULD NOT CLIP HERE
        # acts = tf.clip_by_value(acts, self.action_space.low, self.action_space.high)
        if obs.get_shape().ndims == 1:
            reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=0)
            reward_run = tf.square(obs[8])
        elif obs.get_shape().ndims == 2:
            reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
            reward_run = tf.square(obs[:, 8])
        else:
            raise NotImplementedError
        reward = reward_run + reward_ctrl
        return reward

    def tf_dl(self, obs, act, next_obs):
        """

        :param obs: (obs_dim,)
        :param act: (act_dim,)
        :param next_obs: (obs_dim,)
        :param f_x: (obs_dim, obs_dim)
        :param f_xx: (obs_dim, obs_dim, obs_dim)
        :return:
        """
        # l_x
        e_8 = np.zeros((self.obs_dim))
        e_8[8] = 1
        r_x = 2*obs[8] * e_8
        l_x = -r_x

        # l_u
        r_u = -0.2 * act
        l_u = -r_u

        # l_xx
        r_xx = np.zeros((self.obs_dim, self.obs_dim))
        r_xx[8, 8] = 2
        l_xx = -r_xx

        # l_uu
        r_uu = np.eye(self.act_dim) * (-0.2)
        l_uu = -r_uu

        # l_ux
        l_ux = tf.zeros((self.act_dim, self.obs_dim))

        return l_x, l_u, l_xx, l_uu, l_ux


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