import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import norm

import random
import time
import pickle
import os
import multiprocessing as mp
from copy import deepcopy as copy
from settings import USE_GPU, dtype, device, num_cpu

import utils
# import grapher
# from envs.particle_env_b import ParticleMazeEnv

class Agent():

	def __init__(self, problem_params, env_params):
		self.problem_params = problem_params
		self.env_params = env_params

		# Store handy variables for math
		self.vars = {
			'T': self.problem_params['T'],
			'gamma': self.problem_params['gamma'],
			'buf_size': self.problem_params['buffer_size']
		}

		# Initialize the environment
		if self.problem_params['env_type'] == 'Particle':
			self.env = ParticleMazeEnv(dense=False)
			self.vars['N'], self.vars['M'] = 4, 2
			self.vars['min_act'], self.vars['max_act'] = -1, 1
			self.mujoco = False
		else:
			self.env = gym.make(self.problem_params['env_type'])
			self.vars['N'], self.vars['M'] = self.env.observation_space.shape[0], self.env.action_space.shape[0]
			self.vars['min_act'], self.vars['max_act'] = self.env.action_space.low[0], self.env.action_space.high[0]
			self.mujoco = True

		# Basic information for reinforcement learning
		self.time = 0
		self.buf = utils.ReplayBuffer(self.vars['N'], self.vars['M'], self.vars['buf_size'])

		# Maybe we don't want to start in a fresh environment
		if 'env_copy' in self.problem_params and self.problem_params['env_copy']:
			self.env = self.problem_params['env_copy']
			self.prev_obs = self.env._get_obs() # Not all envs have this
		else:
			self.prev_obs = self.env.reset()

		# Store histories of interactions (outside of replay buffer)
		self.hist = {
			'obs': np.zeros((self.vars['T'], self.vars['N'])),
			'rew': np.zeros(self.vars['T']),
			'done': np.zeros(self.vars['T']),
			'env': [[] for _ in range(self.vars['T'])]
		}
		self.compute_hist = {
			'plan': np.zeros(self.vars['T']),
			'update': np.zeros(self.vars['T'])
		}

		# Default place to store output files
		if not self.problem_params['dir_name']:
			import datetime
			now = datetime.datetime.now()
			ctime = '%02d%02d_%02d%02d' % (now.month, now.day, now.hour, now.minute)
			self.output_dir = 'ex/' + ctime
		else:
			self.output_dir = self.problem_params['dir_name']
		
		# Extra logging information
		self.last_reset = 0
		self.begin_time = time.time()
		self.freeze = self.problem_params['freeze']
		self.cache = ()

		# For use with MPC/other planning
		self.use_best_plan = False
		self.best_plan_action = np.zeros(self.vars['M'])
		self.best_plan_val = -float('inf')

		# Printing information
		print('Initialized %s agent' % self.problem_params['env_type'])
		print(self)
		print('Number of CPUs: %d, torch device:' % num_cpu, device)
		print('Saving to dir: %s' % self.output_dir)
		print('N: %d, M: %d' % (self.vars['N'], self.vars['M']))

	def run_timestep(self):
		if self.problem_params['render_env']:
			print(self.env.render(mode='rgb_array'))

		if self.use_best_plan:
			self.best_plan_action = np.zeros(self.vars['M'])
			self.best_plan_val = -float('inf')

		done_signal = self.update_env()
		if done_signal and self.time > 0:
			self.hist['done'][self.time-1] = 1
			self.buf.buffer['done'][(self.buf.ind-1) % self.buf.size] = 1

		if self.time % self.problem_params['img_freq'] == 0 and self.time > 0:
			if self.problem_params['env_type'] == 'Particle':
				self.save_particle_img_obj()

		check = time.time()
		action = self.get_action()
		if self.use_best_plan:
			action = self.best_plan_action

		self.step(action)
		self.compute_hist['plan'][self.time-1] = time.time() - check

		check = time.time()
		if not self.freeze:
			self.do_updates()
		self.compute_hist['update'][self.time-1] = time.time() - check

		if self.time % self.problem_params['print_freq'] == 0:
			self.print_logs()

		if self.time % self.problem_params['save_freq'] == 0:
			self.save_self()

	def get_action(self):
		raise NotImplementedError

	def do_updates(self):
		pass

	def step(self, action, print_info=False):
		# Execute action in environment
		action = np.clip(action, self.vars['min_act'], self.vars['max_act'])
		obs, rew, done, ifo = self.env.step(action)
		if print_info:
			print('Timestep %d' % self.time)
			print('Environment info:', ifo)
			print('Reward: %.4f, done:' % rew, done)

		# Update buffers
		self.buf.update(self.prev_obs, obs, rew, action, done)
		if not self.problem_params['do_resets']:
			done = False
		self.update_history({'obs': obs, 'rew': rew, 'done': done})
		
		# Reset if desired
		if done and self.problem_params['do_resets']:
			if print_info:
				print('Resetting environment')
			self.prev_obs = self.env.reset()
		else:
			self.prev_obs = obs

		# Keep track of timesteps taken in environment
		self.time += 1

	def eval_traj(self, actions, terminal=None):
		# Evaluate a trajectory in the simulated dynamics model
		env, cum_rew, fin_std = copy(self.env), 0, 0
		if self.mujoco:
			env.sim.set_state(self.env.sim.get_state())
		if 'Hopper' in self.problem_params['env_type']:
			env.set_target_vel(self.env.get_target_vel())
		for t in range(self.vars['H']):
			obs, rew, done, ifo = env.step(actions[t])
			cum_rew += (self.vars['gamma'] ** t) * rew
		emp_rew = cum_rew
		if terminal:
			preds = terminal.get_preds_np(obs)
			cum_rew += (self.vars['gamma'] ** (t+1)) * np.max(preds)
			fin_std = (self.vars['gamma'] ** (t+1)) * np.std(preds)
		return cum_rew, fin_std, emp_rew

	def update_history(self, infos):
		# Update history (outside of replay buffer)
		for key in infos:
			if key in self.hist:
				self.hist[key][self.time] = infos[key]
			else:
				print('WARNING: %s not found in self.hist' % key)

	def update_env(self):
		done_signal = False

		if self.problem_params['ep_len'] and \
		   self.time % self.problem_params['ep_len'] == 0:
		   self.prev_obs = self.env.reset()
		   done_signal = True

		if self.problem_params['env_type'] == 'Particle':
			# Reset upon reaching the goal and sitting there for a while
			if self.problem_params['ep_len'] is None \
			   and np.sum(self.hist['rew'][self.time-25:self.time]) >= 20 \
			   and self.time-self.last_reset > 10:
				self.last_reset = self.time
				self.prev_obs = self.env.reset()
				done_signal = True

			self.env.spawn_mode = self.env_params['spawn_mode']
			self.env.late_spawn_time = self.env_params['late_spawn_time']

			if self.time == 0:
				self.env._reset_grid(self.env_params['orig_grid'])

			if self.time == self.env_params['world_reset']:
				self.env._reset_grid(self.env_params['new_grid'])
				self.prev_obs = self.env.reset()
				done_signal = True

		elif 'Hopper' in self.problem_params['env_type']:
			if self.time % self.env_params['vel_every'] == 0:
				if self.env_params['rand_vel']:
					vmin, vmax = self.env_params['vel_min'], self.env_params['vel_max']
					self.env.set_target_vel(random.random() * (vmax-vmin) + vmin)
				else:
					sch_ind = (self.time // self.env_params['vel_every']) % len(self.env_params['vel_schedule'])
					self.env.set_target_vel(self.env_params['vel_schedule'][sch_ind])

			self.hist['env'][self.time].append(copy(self.env.sim.data.qpos))
			self.hist['env'][self.time].append(copy(self.env.sim.data.body_xpos))
			self.hist['env'][self.time].append(self.env.get_target_vel())

			if self.time % self.problem_params['print_freq'] == 0 and self.time > 0:
				prev_x = self.hist['env'][self.time-self.problem_params['print_freq']][0][0]
				x = self.hist['env'][self.time][0][0]
				vel = (x - prev_x) / (self.problem_params['print_freq'] * self.env.dt)
				print(' %15s: %.5f' % ('current x', x))
				print(' %15s: %.5f' % ('avg vel', vel))
				if self.env.get_target_vel() is not None:
					print(' %15s: %.5f' % ('target vel', self.env.get_target_vel()))

		return done_signal

	def print_logs(self):
		print('=' * 34)
		print(' Timestep %d' % self.time)
		print(' %.2f sec' % (time.time() - self.begin_time))

		bi, ei = self.time - self.problem_params['print_freq'], self.time

		print(' %15s: %.5f' % ('reward avg', np.mean(self.hist['rew'][bi:ei])))
		print(' %15s: %.5f' % ('reward max', np.max(self.hist['rew'][bi:ei])))

		run_len = self.problem_params['ep_len'] if self.problem_params['ep_len'] is not None else 250
		print(' %15s: %.5f' % ('run reward avg', np.mean(self.hist['rew'][max(0,ei-run_len):ei])))
		print(' %15s: %.5f' % ('run reward max', np.max(self.hist['rew'][max(0,ei-run_len):ei])))

		print(' %15s: %.5f' % ('avg plan sec', 
			np.mean(self.compute_hist['plan'][bi:ei])))
		print(' %15s: %.5f' % ('avg update sec', 
			np.mean(self.compute_hist['update'][bi:ei])))

		print(' %15s: %.5f' % ('tot plan sec', 
			np.sum(self.compute_hist['plan'][:ei])))
		print(' %15s: %.5f' % ('tot update sec', 
			np.sum(self.compute_hist['update'][:ei])))

		return bi, ei

	def save_self(self):
		output_file = self.output_dir + '/checkpoints'
		if not os.path.isdir(output_file):
			os.makedirs(output_file)
		file_name = output_file + '/model_' + str(self.time) + '.pkl'
		file = open(file_name, 'wb')
		pickle.dump(self, file)
		file.close()
		print('Saved model to:', file_name)

	def save_particle_img_obj(self):
		pass

	def run_lifetime(self):
		while self.time < self.vars['T']:
			self.run_timestep()
