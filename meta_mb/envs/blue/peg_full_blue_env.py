import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os

class PegInsertion(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(**locals())

        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_full_peg_v1.xml')
        self.goal_peg = np.zeros((3,))
        #self.goal_left = np.zeros((3, ))
        mujoco_env.MujocoEnv.__init__(self, xml_file, 2)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat[:-3],
            #self.ee_position('left') - self.goal_left,
            self.ee_position('right') - self.goal_peg,
        ])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        #vec_left = self.ee_position('left') - self.goal_left,
        vec_right = self.ee_position('right') - self.goal_peg,
        reward_dist = - 0.5 * np.linalg.norm(vec_right)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + 1.25e-4 * reward_ctrl
        observation = self._get_obs()
        done = False
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return observation, reward, done, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.goal_peg = np.random.uniform(low=[0.25, -1.75, 1.25], high=[0.75, -1.25, 1.5])
        #qpos[-6:-3] = self.goal_left
        qpos[-3:] = self.goal_peg
        qvel[-6:] = 0

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def ee_position(self, arm):
        return self.get_body_com(arm + '_r_finger_tip_link')

    def elbow_position(self, arm):
        return None;

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 0
        self.viewer.cam.azimuth = 180

if __name__ == "__main__":
    env = PegInsertion()
    while True:
        env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            env.step(action)
            env.render()