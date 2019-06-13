import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from meta_mb.meta_envs.base import RandomEnv
import os


class FullBlueEnv(RandomEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(**locals())

        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_full_v1.xml')
        self.goal_left = np.zeros((3,))
        self.goal_right = np.zeros((3,))

        RandomEnv.__init__(self, 0, xml_file, 2)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat[:-3],
            self.ee_position('left') - self.goal_left,
            self.ee_position('right') - self.goal_right,
        ])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        vec_left = self.ee_position('left') - self.goal_left,
        vec_right = self.ee_position('right') - self.goal_right
        reward_dist = - 0.5 * (np.linalg.norm(vec_left) + np.linalg.norm(vec_right))
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + 1.25e-4 * reward_ctrl
        observation = self._get_obs()
        done = False
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return observation, reward, done, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.goal_left = np.random.uniform(low=[0.25, 0.25, 0.25], high=[0.75, 0.75, 0.5])
        self.goal_right = np.random.uniform(low=[0.25, -0.75, 0.25], high=[0.75, -0.25, 0.5])
        qpos[-6:-3] = self.goal_left
        qpos[-3:] = self.goal_right
        qvel[-6:] = 0

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def ee_position(self, arm):
        return (self.get_body_com(arm + '_r_finger_tip_link')
                + self.get_body_com(arm + '_l_finger_tip_link'))/2

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            vec_left = self.ee_position('left') - self.goal_left
            vec_right = self.ee_position('right') - self.goal_right
            reward_ctrl = -np.sum(np.square(act), axis=1)
            reward_dist = -0.5 * (np.linalg.norm(vec_right))
            reward = reward_dist + 1.25e-4 * reward_ctrl
            return np.clip(reward, -1e2, 1e2)
        #elif obs.ndim == 1:
        #    assert obs.shape == obs_next.shape
        #    reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act))
        #    reward_run = obs_next[8]
        #    reward = reward_run + reward_ctrl
        #    return np.clip(reward, -1e2, 1e2)
        else:
            raise NotImplementedError

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 0
        self.viewer.cam.azimuth = 180


if __name__ == "__main__":
    env = FullBlueEnv()
    while True:
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)
            env.render()
