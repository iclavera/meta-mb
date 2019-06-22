import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os


class BlueEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, arm='right'):
        utils.EzPickle.__init__(**locals())

        assert arm in ['left', 'right']
        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_' + arm + '_v2.xml')

        self.goal = np.zeros((3,))
        self._arm = arm

        mujoco_env.MujocoEnv.__init__(self, xml_file, 2)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat[:-3],
            self.get_body_com("right_gripper_link"),
            self.ee_position - self.goal,
        ])

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        vec = self.ee_position - self.goal
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + 1.25e-4 * reward_ctrl
        observation = self._get_obs()
        done = False
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return observation, reward, done, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.goal = np.random.uniform(low=[-0.75, -0.25, 0.25], high=[-0.25, 0.25, 0.5])
        #self.goal = np.array([-0.65, -0.2, 0.21]) #fixed goal
        qpos[-3:] = self.goal
        qvel[-3:] = 0

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    @property
    def ee_position(self):
        return (self.get_body_com(self._arm + '_r_finger_tip_link')
                + self.get_body_com(self._arm + '_l_finger_tip_link'))/2

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 0


if __name__ == "__main__":
    env = BlueEnv('right')
    while True:
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)
            env.render()
