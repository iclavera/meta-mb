import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import random
import time
import pickle
import multiprocessing as mp
from copy import deepcopy as copy
from ensemble import MLP, ValEnsemble, PolEnsemble
from settings import USE_GPU, dtype, device, num_cpu

class ReplayBuffer():

	def __init__(self, state_dim, act_dim, size):
		self.buffer = dict()
		self.buffer['state'] = np.zeros((size, state_dim))
		self.buffer['next_state'] = np.zeros((size, state_dim))
		self.buffer['reward'] = np.zeros(size)
		self.buffer['action'] = np.zeros((size, act_dim))
		self.buffer['done'] = np.zeros(size)

		self.ind = 0
		self.total_in = 0
		self.size = size
		self.N = state_dim
		self.M = act_dim

	def get(self, query, ind):
		return self.buffer[query][ind % self.size]

	def advance(self):
		self.ind = (self.ind+1) % self.size
		self.total_in += 1

	def update(self, s, ns, r, a, done=False):
		self.buffer['state'][self.ind] = s
		self.buffer['next_state'][self.ind] = ns
		self.buffer['reward'][self.ind] = r
		self.buffer['action'][self.ind] = a
		self.buffer['done'][self.ind] = 1 if done else 0
		
		self.advance()

	def sample_batch(self, num_samples):
		max_ind = min(self.size, self.total_in)
		inds = np.random.randint(0, max_ind, size=num_samples)

		states = np.zeros((num_samples, self.N))
		next_states = np.zeros((num_samples, self.N))
		actions = np.zeros((num_samples, self.M))
		rewards = np.zeros(num_samples)
		done = np.zeros(num_samples)
		
		for i in range(num_samples):
			states[i] = self.get('state', inds[i])
			next_states[i] = self.get('next_state', inds[i])
			actions[i] = self.get('action', inds[i])
			rewards[i] = self.get('reward', inds[i])
			done[i] = self.get('done', inds[i])

		return states, next_states, actions, rewards, done

def get_traj_targets(buf, ensemble, H, gamma):
	size = min(buf.size, buf.total_in)
	states = np.zeros((size, buf.N))
	actions = np.zeros((size, buf.M))
	cum_rews = np.zeros(size+H+1)
	targets = np.zeros(size)
	last_done = size+H+1

	for i in reversed(range(size, size+H)):
		cum_rews[i] = buf.get('reward', i) + gamma * cum_rews[i+1]
		if buf.get('done', i):
			last_done = i

	for i in reversed(range(size)):
		states[i] = buf.get('state', i)
		actions[i] = buf.get('action', i)
		cum_rews[i] = buf.get('reward', i) + gamma * cum_rews[i+1]
		if buf.get('done', i):
			last_done = i

		max_k = min([H, buf.total_in-i, last_done-i])
		fs = buf.get('state', i+max_k-1)
		targets[i] = cum_rews[i] + (gamma ** max_k) * (ensemble.get_value(fs).item() - cum_rews[i+max_k])

	"""
	for i in range(size):
		states[i] = buf.get('state', i)
		dis, max_k = 1, min(H, buf.total_in-i)
		for k in range(max_k):
			targets[i] += dis * buf.get('reward', i+k)
			dis *= gamma
		fs = buf.get('next_state', i+max_k-1)
		targets[i] += dis * ensemble.get_value(fs).item()
	"""

	return states, actions, targets

def get_act_targets(buf):
	size = min(buf.size, buf.total_in)
	states = np.zeros((size, buf.N))
	actions = np.zeros((size, buf.M))

	for i in range(size):
		states[i] = buf.get('state', i)
		actions[i] = buf.get('action', i)

	return states, actions

def get_dis_rew(rews, gamma):
	rew, dis = 0, 1
	for r in rews:
		rew += dis * r
		dis *= gamma
	return rew

def get_rollouts(start_env, actions, N, do_resets=False):
	num_rollouts, H, M = actions.shape
	states, rews = np.zeros((num_rollouts, H, N)), np.zeros((num_rollouts, H))
	dones = np.zeros((num_rollouts, H))

	for i in range(num_rollouts):
		env = copy(start_env)
		for k in range(H):
			obs, rew, done, _ = env.step(actions[i,k])
			states[i,k], rews[i,k] = obs, rew
			dones[i,k] = done
			# if done and do_resets:
			# 	obs = env.reset()
			# if done:
			# 	break

	return states, rews, dones

def get_rollouts_star(args_list):
	return get_rollouts(*args_list)

def get_rollouts_mp(start_env, actions, N, do_resets=False):
	if num_cpu == 1:
		return get_rollouts(start_env, actions, N)
	else:
		per_cpu = actions.shape[0] // num_cpu
		pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
		parallel_runs = [
			pool.apply_async(get_rollouts_star, 
			args=([start_env, actions[i*per_cpu:(i+1)*per_cpu], N],))
			for i in range(num_cpu)
		]

		try:
			results = [p.get(timeout=360000) for p in parallel_runs]
		except Exception as e:
			print(str(e))
			print('Timeout error raised, trying again')

			pool.close()
			pool.terminate()
			pool.join()

			return get_rollouts_mp(start_env, actions, N)

		pool.close()
		pool.terminate()
		pool.join()

		num_rollouts, H, M = actions.shape
		states = np.zeros((num_rollouts, H, N))
		rews = np.zeros((num_rollouts, H))
		dones = np.zeros((num_rollouts, H))

		for i in range(num_cpu):
			s, r, ds = results[i]
			beg_ind, end_ind = i*per_cpu, (i+1)*per_cpu
			states[beg_ind:end_ind,:,:] = s
			rews[beg_ind:end_ind,:] = r
			dones[beg_ind:end_ind] = ds

		return states, rews, dones

def pi2(start_env, ensemble, num_paths, H, N, M, gamma,
		min_act, max_act, prior, mu, covar):
	raw_eps = np.random.multivariate_normal(np.zeros(M), covar, H*num_paths)
	eps = np.zeros((num_paths, H, M))
	for i in range(H*num_paths):
		eps[i//H,i%H] = raw_eps[i]

	actions = np.tile(prior+mu, (num_paths, 1, 1)) + eps
	actions = np.clip(actions, min_act, max_act)

	states, rews, dones = get_rollouts_mp(start_env, actions, N)

	R = np.zeros(num_paths)
	for i in range(num_paths):
		l_s = states[i,0]
		for k in range(H):
			l_s = states[i,k]
			if dones[i,k]:
				break

		R[i] = get_dis_rew(rews[i], gamma)
		if ensemble is not None:
			V_hat = ensemble.get_value(l_s)
			R[i] += (gamma ** (k+1)) * V_hat.item()

	return actions, states, rews, R, dones, eps

def generate_rollouts(start_env, planned_actions, ensemble, num_paths, N, gamma,
					  mppi_mean, mppi_var, min_act, max_act, do_resets,
					  cpu_num=None, prior_actions=None):
	H, M = planned_actions.shape
	actions = np.zeros((num_paths, H, M))

	for i in range(num_paths):
		eps = np.random.normal(mppi_mean, mppi_var, planned_actions.shape)
		for t in range(2, eps.shape[0]):
			eps[t] = 0.05*eps[t] + .8*eps[t-1]
		actions[i] = planned_actions + eps
		# if prior_actions is None:
		# 	actions[i] = planned_actions + eps
		# else:
		# 	actions[i] = prior_actions[i] + eps
	actions = np.clip(actions, min_act, max_act)
	states, rews, dones = get_rollouts(start_env, actions, N, do_resets)

	R = np.zeros(num_paths)
	for i in range(num_paths):
		# l_s = states[i,0]
		# for k in range(H):
		# 	l_s = states[i,k]
		# 	if dones[i,k]:
		# 		break
		
		R[i] = get_dis_rew(rews[i], gamma)
		if ensemble is not None:
			l_s = states[i,-1]
			V_hat = ensemble.get_value(l_s)
			R[i] += (gamma ** H) * V_hat.item()

	return actions, states, rews, R, dones

def generate_rollouts_star(args_list):
	res = generate_rollouts(*args_list)
	return res

def generate_rollouts_mp(start_env, planned_actions, ensemble, num_paths, N, gamma,
						 mppi_mean, mppi_var, min_act, max_act, do_resets,
						 num_cpu=1, prior_actions=None):
	H, M = planned_actions.shape
	paths_per_cpu = num_paths // num_cpu
	num_rollouts = paths_per_cpu * num_cpu
	args_list = [planned_actions, ensemble, paths_per_cpu, N, gamma,
				 mppi_mean, mppi_var, min_act, max_act, do_resets]
	
	if num_cpu == 1:
		actions, states, rews, R, dones = generate_rollouts_star([start_env]+args_list+[0,prior_actions])
	else:
		pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
		parallel_runs = [pool.apply_async(generate_rollouts_star, 
						 args=([copy(start_env)]+args_list+[i,prior_actions],)) for i in range(num_cpu)]
		try:
			results = [p.get(timeout=360000) for p in parallel_runs]
		except Exception as e:
			print(e)
			print('Timeout error raised, trying again')

			pool.close()
			pool.terminate()
			pool.join()

			return generate_rollouts_mp(start_env, planned_actions, ensemble, num_paths, N,
										gamma, mppi_mean, mppi_var, min_act, max_act,
										do_resets, num_cpu, prior_actions)

		pool.close()
		pool.terminate()
		pool.join()

		actions = np.zeros((num_rollouts, H, M))
		states = np.zeros((num_rollouts, H, N))
		rews = np.zeros((num_rollouts, H))
		R = np.zeros(num_rollouts)
		dones = np.zeros((num_rollouts, H))

		for i in range(num_cpu):
			a, s, r, rr, ds = results[i]
			beg_ind, end_ind = i*paths_per_cpu, (i+1)*paths_per_cpu
			actions[beg_ind:end_ind,:,:] = a
			states[beg_ind:end_ind,:,:] = s
			rews[beg_ind:end_ind,:] = r
			R[beg_ind:end_ind] = rr
			dones[beg_ind:end_ind] = ds

	return actions, states, rews, R, dones

def load_agent(file_name):
	agent_file = open(file_name, 'rb')
	agent = pickle.load(agent_file)
	agent_file.close()
	return agent
