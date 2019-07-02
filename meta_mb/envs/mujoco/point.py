from meta_mb.meta_envs.base import MetaEnv
import os
from gym.envs.mujoco.mujoco_env import MujocoEnv

import numpy as np
from gym import utils

class PointEnv(MetaEnv, MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'point2.xml'), 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        dist = np.linalg.norm(ob[:2] - np.array([2, 2]))
        done =  dist < 0.1
        reward = -dist
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        # v = self.viewer
        # v.cam.trackbodyid = 0
        # v.cam.distance = self.model.stat.extent
        pass


if __name__ == "__main__":
    env = PointEnv()
    env.reset()
    for _ in range(1000):
        env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
