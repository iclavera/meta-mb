from meta_mb.meta_envs.base import MetaEnv
import os
from gym.envs.mujoco.mujoco_env import MujocoEnv

import numpy as np
from gym import utils

class PointEnv(MetaEnv, MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'point_pos.xml'), 2)

    def step(self, a):
        desired_pos = self.get_xy() + np.clip(a, -20, 20) / 50.
        for _ in range(40):
            self.do_simulation(desired_pos, self.frame_skip)
            if np.linalg.norm(desired_pos - self.get_xy()) < 0.01 and np.linalg.norm(
                    self.sim.data.qvel.ravel()) < 1e-3:
                break

        ob = self._get_obs()
        dist = np.linalg.norm(ob[:2] - np.array([2, 2]))
        done =  False
        reward = -dist
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

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
