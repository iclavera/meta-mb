import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from meta_mb.meta_envs.base import RandomEnv
import os
from meta_mb.envs.blue.full_blue_env import FullBlueEnv
import time

class MimicBlueActionEnv(FullBlueEnv):

	def __init__(self, parent):
		FullBlueEnv.__init__(self)

	def do_simulation(self, action, frame_skip):
		action = np.clip(action, self._low, self._high)
		assert frame_skip > 0
		for _ in range(frame_skip):
			time.sleep(self.dt)
			qvel = self.sim.data.qvel
			qvel[:-3] = action
			self.set_state(self.sim.data.qpos, qvel)

	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat,
			self.sim.data.qvel.flat[:-3],
			self.sim.data.body_xpos.flat[:3],
			self.ee_position('right') - self.goal_right,
		])
