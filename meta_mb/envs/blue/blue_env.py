import numpy as np
from meta_mb.logger import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
import os


class BlueReacherEnv(MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        self.goal = np.ones((3,))
        MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), "assets", "blue_right_v2.xml"), 2)
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        vec = self.get_body_com("right_l_finger_tip_link") - self.goal
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + 0.5 * 0.01 * reward_ctrl
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = np.random.uniform(low=-.2, high=.2, size=2)
            self.goal = np.ones((3,)) * -.3
            if np.linalg.norm(self.goal) < 2:
                break
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com("right_l_finger_tip_link") - self.goal
        ])


if __name__ == "__main__":
    env = BlueReacherEnv()
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()