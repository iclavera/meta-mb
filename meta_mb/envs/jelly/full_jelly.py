import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os

class FullJellyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(**locals())

        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'jelly.xml')
        # Goal still to be specified
        #self.goal = np.zeros(3)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 2)

    def _get_obs(self):
        return

    def step(self):
        return

    def reset_model(self):
        return

    def foot_position(self, foot):
        return

    def viewer_setup(self):
        return

if __name__ == "__main__":
    env = FullJellyEnv()
    while True:
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)
            env.render()


