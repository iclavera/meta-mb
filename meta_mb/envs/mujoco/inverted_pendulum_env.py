import numpy as np
from mb_vision.logger import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
from mb_vision.envs.normalized_env import normalize

import os
FILE = os.path.join(os.path.dirname(__file__), "assets", "inverted_pendulum.xml")


class InvertedPendulumEnv(MujocoEnv, gym.utils.EzPickle):
    PENDULUM_LENGTH = 0.6
    def __init__(self, store_images=False):
        self._store_images = store_images
        MujocoEnv.__init__(self, FILE, 2)
        gym.utils.EzPickle.__init__(self, store_images)
        self._obs_bounds()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        if self._store_images:
            rgb_img = self.render('rgb_array', width=64, height=64)
        else:
            rgb_img = []
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        # reward = float(notdone)
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, InvertedPendulumEnv.PENDULUM_LENGTH]))) / (
                        InvertedPendulumEnv.PENDULUM_LENGTH ** 2)
        )
        reward -= 0.01 * np.sum(np.square(action))
        done = False
        return ob, reward, done, dict(rgb_img=rgb_img)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            # self.sim.data.qvel.flat
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def reward(self, obs, action, obs_next):
        if obs_next.ndim == 2:
            reward = np.exp(
                -np.sum(np.square(self._get_ee_pos(obs_next) - np.array([[0.0, InvertedPendulumEnv.PENDULUM_LENGTH]])), axis=1) / (
                            InvertedPendulumEnv.PENDULUM_LENGTH ** 2)
            )
            reward -= 0.01 * np.sum(np.square(action), axis=1)
            return reward
        else:
            reward = np.exp(
                -np.sum(np.square(self._get_ee_pos(obs_next) - np.array([0.0, InvertedPendulumEnv.PENDULUM_LENGTH]))) / (
                            InvertedPendulumEnv.PENDULUM_LENGTH ** 2)
            )
            reward -= 0.01 * np.sum(np.square(action))
            return reward

    def _obs_bounds(self):
        jnt_range = self.model.jnt_range
        jnt_limited = self.model.jnt_limited
        self._obs_lower_bounds = -1000 * np.ones(shape=(self.sim.data.qpos.shape[0] + self.sim.data.qvel.shape[0]-1,))
        self._obs_upper_bounds = 1000 * np.ones(shape=(self.sim.data.qpos.shape[0] + self.sim.data.qvel.shape[0]-1,))
        for idx, limited in enumerate(jnt_limited):
            if idx > 0 and limited:
                self._obs_lower_bounds[idx-1] = jnt_range[idx][0]
                self._obs_upper_bounds[idx-1] = jnt_range[idx][1]

    @property
    def obs_lower_bounds(self):
        return self._obs_lower_bounds

    @property
    def obs_upper_bounds(self):
        return self._obs_upper_bounds

    def get_param_values(self):
        return None

    def log_diagnostics(self, paths, prefix=''):
        pass

    @staticmethod
    def _get_ee_pos(x):
        if x.ndim == 1:
            x0, theta = x[0], x[1]
            return np.array([
                x0 - InvertedPendulumEnv.PENDULUM_LENGTH * np.sin(theta),
                InvertedPendulumEnv.PENDULUM_LENGTH * np.cos(theta)
            ])
        elif x.ndim == 2:
            x0, theta = x[:, 0], x[:, 1]
            return np.array([
                x0 - InvertedPendulumEnv.PENDULUM_LENGTH * np.sin(theta),
                InvertedPendulumEnv.PENDULUM_LENGTH * np.cos(theta)
            ]).T


if __name__ == "__main__":
    env = InvertedPendulumEnv()
    env.reset()
    for _ in range(1000):
        env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
