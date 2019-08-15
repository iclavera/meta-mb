import numpy as np 
from gym.envs.mujoco import mujoco_env
from gym import utils
from meta_mb.meta_envs.base import RandomEnv
import os
from meta_mb.envs.blue.blue_env import BlueEnv
import time

class MimicBluePosEnv(BlueEnv): 

	def __init__(self, max_path_len, positions=None):
		self.max_path_len = max_path_len
		self.path_len = 0
		if positions is not None:
			self.positions = positions
		BlueEnv.__init__(self)

	def step(self, act):
        act = np.clip(act, self._low, self._high)
        self.do_simulation(act, self.frame_skip)

        norm = np.linalg.norm(self.get_body_com("right_gripper_link") - self.goal)
        joint_vel = self.sim.data.qvel[:-3]

        reward_ctrl = -self.ctrl_penalty * np.square(np.linalg.norm(act))
        reward_vel = -self.vel_penalty * np.square(np.linalg.norm(joint_vel))
        reward_dist = -(np.square(norm) + np.log(np.square(norm) + self.alpha))

        reward = reward_dist + reward_ctrl + reward_vel

        observation = self._get_obs()
        info = dict(dist=norm, reward_dist=reward_dist, reward_ctrl=reward_ctrl, reward_vel=reward_vel)
        done = False
        if (self.path_len == self.max_path_len):
			done = True
		else:
			done = False
        return observation, reward, done, info

	def do_simulation(self, action, frame_skip):
		action = np.clip(action, self._low, self._high)
		assert frame_skip > 0
		if len(self.positions) != 0:
			position = self.positions[self.path_len]
		for _ in range(frame_skip):
			#time.sleep(self.dt)
			qpos = np.concatenate((position[0], self.goal))
			qvel = np.concatenate((position[1], np.zeros(3)))
			self.set_state(qpos, qvel)
		self.path_len += 1

	def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:-3],
            self.sim.data.qvel.flat[:-3],
            self.get_body_com("right_gripper_link"),
        ])

if __name__ == "__main__":
    env = MimicBluePosEnv(max_path_len=200)
    while True:
        env.reset()
        for _ in range(200):
            env.step(env.action_space.sample())
            env.render()