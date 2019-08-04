from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
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

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        reward_ctrl = -0.1 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, 8]
        reward = reward_run + reward_ctrl
        return reward

    def tf_reward(self, obs, acts, next_obs):
        reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
        reward_run = next_obs[:, 8]
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

    def tf_termination_fn(self, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        done = tf.tile(tf.constant([False]), [tf.shape(obs)[0]])
        done = done[:,None]
        return done

    def termination_fn(self, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        done = np.array([False]).repeat(obs.shape[0])
        done = done[:,None]
        return done



if __name__ == "__main__":
    env = HalfCheetahEnv()
    env.reset()
    for _ in range(1000):
        _ = env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
