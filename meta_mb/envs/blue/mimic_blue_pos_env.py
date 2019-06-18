import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from meta_mb.meta_envs.base import RandomEnv
import os
from meta_mb.envs.blue.full_blue_env import FullBlueEnv
import time

class MimicBluePosEnv(FullBlueEnv):

	def __init__(self, max_path_len, parent=None, positions=None):
		self.max_path_len = max_path_len
		self.parent = parent
		self.curr_step = 0
		if positions is not None:
			self.positions = positions
		FullBlueEnv.__init__(self)
		self.goal_right = self.parent.goal

	def step(self, action):
		self.sim.model.body_pos[-1] = self.parent.goal
		self.do_simulation(action, self.frame_skip)
		vec_right = self.ee_position('right') - self.goal_right
		reward_dist = -np.linalg.norm(vec_right)
		reward_ctrl = -np.square(action/(2* self._high)).sum()
		reward = reward_dist + 0.5 * 0.1 * reward_ctrl
		observation = self._get_obs()
		if (self.curr_step == self.max_path_len):
			done = True
		else:
			done = False
		info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
		return observation, reward, done, info

	def do_simulation(self, action, frame_skip):
		action = np.clip(action, self._low, self._high)
		assert frame_skip > 0
		if len(self.positions) != 0:
			position = self.positions[self.curr_step]
		for _ in range(frame_skip):
			time.sleep(self.dt)
			qpos = np.concatenate((position[0], self.goal_right))
			qvel = np.concatenate((position[1], np.zeros(3)))
			self.set_state(qpos, qvel)
		self.curr_step += 1

	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat,
			self.sim.data.qvel.flat[:-3],
			self.sim.data.body_xpos.flat[:3],
			self.ee_position('right') - self.goal_right,
		])

if __name__ == "__main__":
    env = MimicBluePosEnv(positions=None)
    while True:
        env.reset()
        for _ in range(100):
            env.step(env.action_space.sample())
            env.render()