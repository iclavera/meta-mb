from meta_mb.meta_envs.base import MetaEnv
import os
from gym.envs.mujoco.mujoco_env import MujocoEnv

import numpy as np
from gym import utils
import tensorflow as tf

class PointEnv(MetaEnv, MujocoEnv, utils.EzPickle):
    def __init__(self, random_reset=True):
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'point_pos.xml'), 2)

        self.random_reset = random_reset

    def step(self, a):
        desired_pos = self.get_xy() + np.clip(a, -20, 20) / 30.
        # for _ in range(40):
        #     self.do_simulation(desired_pos, self.frame_skip)
        #     if np.linalg.norm(desired_pos - self.get_xy()) < 0.01 and np.linalg.norm(
        #             self.sim.data.qvel.ravel()) < 1e-3:
        #         break
        desired_pos = np.clip(desired_pos, -2.8, 2.8)
        self.reset_model(pos=desired_pos)

        ob = self._get_obs()
        done =  False
        return ob, self.reward(None, None, ob), done, {}

    def reset_model(self, pos=None):
        if pos is None:
            if self.random_reset:
                qpos = self.init_qpos + np.random.uniform(size=self.model.nq, low=-2., high=2.)
            else:
                qpos = self.init_qpos + np.random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        else:
            qpos = self.init_qpos + pos
        qvel = self.init_qvel + np.random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reward(self, obs, act, obs_next):
        return -np.linalg.norm(obs_next[:2] - np.array([2, 2]))

    def tf_reward(self, obs, act, obs_next):
        return - tf.norm(obs_next[..., :2] - np.array([2, 2]), axis = -1)

    def _get_obs(self):
        return self.sim.data.qpos.ravel()

    def viewer_setup(self):
        # v = self.viewer
        # v.cam.trackbodyid = 0
        # v.cam.distance = self.model.stat.extent
        pass

    def render(self, mode='human', width=100, height=100):
        if mode == 'human':
            super().render(mode=mode)
        else:
            data = self.sim.render(width, height, camera_name='main')
            return data[::-1, :, :]

    def get_xy(self):
        return self.sim.data.qpos.ravel()


if __name__ == "__main__":
    env = PointEnv()
    env.reset()
    for _ in range(1000):
        # import pdb; pdb.set_trace()
        env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
