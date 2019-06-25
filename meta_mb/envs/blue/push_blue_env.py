import numpy as np
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import RandomEnv
from gym import utils
import os

class PushArmBlueEnv(mujoco_env.MujocoEnv, utils.EzPickle):
	def __init__(self, arm='right', log_rand=0):
		utils.EzPickle.__init__(**locals())

		self.goal_obj = np.zeros((3,)) #object to be grabbed
		self.goal_dest = np.zeros((3,)) #destionation to push object

		self._arm = arm

		self.holding_obj = False

		max_torques = np.array([5, 5, 4, 4, 3, 2, 1])
		self._low = -max_torques
		self._high = max_torques

		#assert arm in ['left', 'right']
		xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_push_' + arm + '_v2.xml')

		mujoco_env.MujocoEnv.__init__(self, xml_file, 2)

	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat,
			self.sim.data.qvel.flat[:-3],
			self.get_body_com("right_gripper_link"),
			self.ee_position - self.goal_obj])

	def step(self, act):
		done = False
		if (hasattr(self, "actions")):
			act = self.actions[self.path_len]
			self.path_len += 1
			if(self.path_len == self.max_path_len):
				done = True

		self.do_simulation(act, self.frame_skip)
		#self.correction() # Use for v2 arms
		vec = self.ee_position - self.goal_obj
		reward_dist = -np.linalg.norm(vec)
		reward_ctrl = -np.square(act/(2 * self._high)).sum()
		reward = reward_dist + 0.5 * 0.1 * reward_ctrl
		observation = self._get_obs()
		info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
		return observation, reward, done, info
		"""
		if not self.holding_obj:
						return
		else:
						return 
		"""

	def reward(self, obs, act, obs_next):
		assert obs.ndim == act.ndim == obs_next.ndim
		if obs.ndim == 2:
			assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
			reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act / (2 * self._high)), axis=1)
			reward_dist = -np.linalg.norm(obs_next[:, -3:], axis=1)
			reward = reward_dist + reward_ctrl
			return np.clip(reward, -1e2, 1e2)

		elif obs.ndim == 1:
			return self.reward(obs[None], act[None], obs_next[None])[0]

		else:
			raise NotImplementedError
		"""
		if not self.holding_obj:
			return
		else:
			return 
		"""

	def reset_model(self):
		qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
		qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
		self.goal_obj = np.random.uniform(low=[-0.75, -0.25, 0.05], high=[-0.25, 0.25, 0.05])
		#self.goal = np.array([-0.65, -0.2, 0.21]) #fixed goal
		qpos[-3:] = self.goal_obj
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
		for _ in range(1000):
			action = env.action_space.sample()
			env.step(action)
			env.render()