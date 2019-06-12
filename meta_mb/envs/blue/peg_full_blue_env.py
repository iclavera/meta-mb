import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os
from scipy.spatial.distance import euclidean

class PegFullBlueEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, goal_dist=3e-2):
        utils.EzPickle.__init__(**locals())

        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_full_peg_v1.xml')

        print(np.zeros((3,)))

        self.top_goal = np.array([0.0, 0.3, -0.4])
        self.center_goal = np.array([0.0, 0.3, -0.55])
        self.bottom_goal= np.array([0.0, 0.3, -0.7])

        self.goal_dist = goal_dist # permissible distance from goal

        mujoco_env.MujocoEnv.__init__(self, xml_file, 2)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat[:-3],
            self.peg_location() - self.center_goal
        ])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        reward_dist = - 0.5 * self.peg_dist()
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + 1.25e-4 * reward_ctrl
        observation = self._get_obs()
        done = False
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return observation, reward, done, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.sim.model.body_pos[-8] = np.random.uniform(low=[-0.75, -0.5, 0], high=[0.75, -1, 0.8])
        #self.center_goal = np.random.uniform(low=[1.25, -1.75, 1.25], high=[2.75, -2.25, 2.5])
        qpos[-6:-3] = np.zeros((3,))
        qpos[-3:] = self.center_goal
        qvel[-6:] = 0

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        print(obs.ndim)
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

    def peg_dist(self):
        peg_location = self.peg_location()
        distance = (euclidean(peg_location, self.top_goal)
                    + euclidean(peg_location, self.center_goal)
                    + euclidean(peg_location, self.bottom_goal))
        return distance

    def peg_location(self):
        return self.get_body_com("peg")

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 0
        self.viewer.cam.azimuth = 180

if __name__ == "__main__":
    env = PegFullBlueEnv()
    while True:
        env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            env.step(action)
            env.render()