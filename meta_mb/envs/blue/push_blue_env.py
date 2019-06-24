import numpy as np
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import RandomEnv
from gym import utils
import os

class PushArmBlueEnv(mujoco_env.MujocoEnv, utils.EzPickle):
	def __init__(self, arm='right', log_rand=0):
		utils.EzPickle.__init__(**locals())

		self.goal_obj = np.zeros(3) #object to be grabbed
		self.goal_dest = np.zeros(3) #destionation to push object

		self.holding_obj = False

        assert arm in ['left', 'right']
        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_push_' + arm + '_v2.xml')

        mujoco_env.MujocoEnv.__init__(self, xml_file, 2)

    def _get_obs(self):
    	return np.array([
    		self.sim.data.qpos.flat,
    		self.sim.data.qvel.flat,
    		self.get_body_com("right_gripper_link"),
    		self.ee_position - self.goal_dest]) #add more as needed

    def step(self, act):
    	if not self.holding_obj:
    		return
    	else:
			return 

    def reward(self, obs, act, obs_next):
    	if not self.holding_obj:
    		return
    	else:
    		return 

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
    env = PushArmBlueEnv('right')
    while True:
        env.reset()
        for _ in range(200):
            action = env.action_space.sample()
            env.step(action)
            env.render()