import numpy as np
#from gym.envs.mujoco import mujoco_env
#from gym import utils
import os
import gym
from meta_mb.logger import logger
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv

class FullJellyEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'jelly.xml')

        self.goal = np.append(np.random.uniform(-5, 5, 2), np.random.uniform(0, 0.15))
        MujocoEnv.__init__(self, xml_file, 2)
        gym.utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.body_position() - self.goal
        ])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        vec_to_goal = self.body_position() - self.goal
        reward_dist = -0.5*(np.linalg.norm(vec_to_goal))
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + 1.25e-4 *reward_ctrl
        observation = self._get_obs()
        done = False
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return observation, reward, done, info

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act), axis=1)
            reward_run = obs_next[:, 8]
            reward = reward_run + reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        elif obs.ndim == 1:
            assert obs.shape == obs_next.shape
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act))
            reward_run = obs_next[8]
            reward = reward_run + reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        else:
            raise NotImplementedError

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.sim.model.body_pos[-1] = np.append(np.random.uniform(-5, 5, 2), np.random.uniform(0, 0.15))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def body_position(self):
        return self.get_body_com("base_link")

    def foot_position(self, foot):
        return

    def viewer_setup(self):
        return

if __name__ == "__main__":
    env = FullJellyEnv()
    while True:
        env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            env.step(action)
            env.render()


