import numpy as np
import time


class GroundTruthDynamics:
	def __init__(self, envs):
		self.envs = envs
		self.predict_mode = 'serial'

	def predict(self, obs, act, **kwargs):
		if self.predict_mode == 'serial':
			return self.serial_predict(obs, act)



	def serial_predict(self, obs, act):
		start = time.time()
		ret = np.zeros_like(obs)
		for i, o in enumerate(obs):
			self.envs[i].reset_from_obs(o)
			ret[i] = self.envs[i].step(act[i])[0]
		# print(time.time() - start)
		return ret

	def reset_from_obs(self, observations):
		[env.reset_from_obs(o) for (env, o) in zip(self.envs, observations)]
