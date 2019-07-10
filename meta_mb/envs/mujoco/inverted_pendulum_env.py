from meta_mb.meta_envs.base import MetaEnv
import os
from gym.envs.mujoco.mujoco_env import MujocoEnv
import tensorflow as tf

import numpy as np
from gym import utils

class InvertedPendulumEnv(MetaEnv, MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        # done = not notdone
        # reward = int(notdone)
        done = False
        reward = self._get_reward()
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def _get_reward(self):
        old_ob = self._get_obs()
        reward = -((old_ob[1]) ** 2)
        return reward

    def viewer_setup(self):
        v = self.viewer
        v.cam.fixedcamid = 0
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 1.5

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        return -(obs[:, 1]) ** 2

    def tf_reward(self, obs, acts, next_obs):
        return - tf.square(obs[:, 1])



if __name__ == "__main__":
    env = InvertedPendulumEnv()
    # env.reset()
    # for _ in range(1000):
    #     env.render()
    #     ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
    # env.render()
    # env.viewer_setup()
    import matplotlib.pyplot as plt
    k = 1
    plt.figure(figsize=(60, 9))
    for i in range(3):
        env.reset()
        for __ in range(20):
            img = env.render(mode='rgb_array')
            plt.subplot(3, 20, k)
            k += 1
            plt.imshow(img)
            # env.render()
            env.step(env.action_space.sample())

    plt.savefig('invertedpendelum_traj.png')