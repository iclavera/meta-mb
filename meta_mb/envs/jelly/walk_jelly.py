import numpy as np
#from gym.envs.mujoco import mujoco_env
#from gym import utils
import os
import gym
from meta_mb.logger import logger
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv
from meta_mb.meta_envs.base import RandomEnv

class WalkJellyEnv(RandomEnv, gym.utils.EzPickle):
    def __init__(self):
        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'jelly.xml')

        RandomEnv.__init__(self, 0,  xml_file, 2)
        gym.utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.sim.data.body_xpos.flat[:3],
            self.body_position()
        ])

    def step(self, action):
        self.prev_pos = self.body_position()
        self.do_simulation(action, self.frame_skip)
        self.curr_pos = self.body_position()

        reward_dist = -(np.linalg.norm(self.curr_pos - self.prev_pos)) / self.dt
        reward_ctrl = -0.5*0.1*np.square(action).sum()
        reward = reward_dist - reward_ctrl + 1

        observation = self._get_obs()
        done = False
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return observation, reward, done, info

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            reward_dist = -(np.linalg.norm(self.curr_pos - self.prev_pos)) / self.dt
            reward_ctrl = -0.5*0.1*np.square(act).sum()
            reward = reward_dist - reward_ctrl + 1
            return reward

        elif obs.ndim == 1:
            return self.reward(obs[None], act[None], obs_next[None])[0]

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

if __name__ == "__main__":
    env = WalkJellyEnv()
    while True:
        env.reset()
        for _ in range(10000):
            action = env.action_space.sample()
            env.step(action)
            env.render()