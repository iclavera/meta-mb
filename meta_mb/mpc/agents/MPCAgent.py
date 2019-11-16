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
from agents.base_agent import Agent

class MPCAgent(Agent):

	def __init__(self, problem_params, env_params, mpc_params):
		super(MPCAgent, self).__init__(problem_params, env_params)
		self.mpc_params = mpc_params

		self.vars['H'] = self.mpc_params['H']
		self.vars['num_rollouts'] = self.mpc_params['num_rollouts']
		self.vars['mpc_steps'] = self.mpc_params['mpc_steps']
		self.vars['filter_coefs'] = self.mpc_params['filter_coefs']
		self.vars['mppi_temp'] = self.mpc_params['mppi_temp']

		self.mode = self.mpc_params['mode']

		self.planned_actions = np.zeros((self.vars['H'], self.vars['M']))

		self.use_best_plan = self.mpc_params['use_best_plan']

		self.planner_polyak = self.mpc_params['planner_polyak']
		self.planner_weight = 1

		self.hist['plan'] = [[] for _ in range(self.vars['T'])]

	def get_action(self, terminal=None, prior=None):
		if prior is not None:
			pct = self.planner_weight
			prior = (1-pct) * prior + pct * self.planned_actions
			self.planned_actions = prior
			self.planner_weight *= self.planner_polyak
		for i in range(self.vars['mpc_steps']):
			if i == 0:
				self.update_plan(terminal=terminal, prior=prior)
			else:
				self.update_plan(terminal=terminal, prior=self.planned_actions)

		return self.advance_plan()

	def print_logs(self):
		bi, ei = super(MPCAgent, self).print_logs()

		num_planning_iters = sum(len(self.hist['plan'][i]) for i in range(bi, ei))
		print(' %15s: %.5f' % ('avg planning', num_planning_iters / (ei-bi)))

		if len(self.cache) >= 1:
			R = self.cache[1]
			print(' %15s: %.5f' % ('pred plan max', np.max(R)))
			print(' %15s: %.5f' % ('pred plan avg', np.mean(R)))
			print(' %15s: %.5f' % ('pred plan std', np.std(R)))

		if len(self.hist['plan'][self.time-1]) > 0:
			print(' %15s: %.5f' % ('pred mpc avg', self.hist['plan'][self.time-1][-1][2]))

		return bi, ei

	def update_plan(self, num_rollouts=None, terminal=None, prior=None):
		if num_rollouts is None:
			num_rollouts = self.vars['num_rollouts']
		if prior is None:
			prior = self.planned_actions

		paths = self.get_rollouts(prior, num_rollouts)

		actions = np.zeros((len(paths), self.vars['H'], self.vars['M']))
		states = np.zeros((len(paths), self.vars['H'], self.vars['N']))
		rews = np.zeros((len(paths), self.vars['H']))
		dones = np.zeros((len(paths), self.vars['H']))
		for i in range(len(paths)):
			actions[i] = paths[i]['act']
			states[i] = paths[i]['obs']
			rews[i] = paths[i]['rew']
			dones[i] = paths[i]['done']

		R, emp_R = self.score_trajectories(paths, terminal)

		if self.use_best_plan:
			i = np.argmax(R)
			if R[i] > self.best_plan_val:
				self.best_plan_action = actions[i,0]
				self.best_plan_val = R[i]

		if self.mode == 'mppi':
			self.combine_actions_mppi(actions, R)
		elif self.mode == 'best_act':
			best_act_ind = np.argmax(R)
			self.planned_actions = actions[best_act_ind]
		else:
			print('WARNING: %s not registered as an MPC mode' % self.mode)

		# Logging planning information
		ro_mean, ro_std = np.mean(R), np.std(R)
		emp_mean, emp_std = np.mean(emp_R), np.std(emp_R)
		fin_rew, fin_std, emp_rew = self.eval_traj(self.planned_actions, terminal)
		self.hist['plan'][self.time].append(
			[ro_mean, ro_std, fin_rew, fin_std, emp_mean, emp_std, emp_rew, self.best_plan_val]
		)

		if self.use_best_plan and fin_rew > self.best_plan_val:
			self.best_plan_action = self.planned_actions[0]
			self.best_plan_val = fin_rew

		self.cache = (states, R, dones)

		return states, R, dones

	def combine_actions_mppi(self, actions, R):
		advs = R - np.max(R) # (R - np.min(R)) / (np.max(R) - np.min(R) + 1e-6)
		S = np.exp(advs / self.vars['mppi_temp'])
		weighted_seq = S * actions.T
		self.planned_actions = np.sum(weighted_seq.T, axis=0) / (np.sum(S) + 1e-6)

	def score_trajectories(self, paths, terminal):
		scores, raw_scores = np.zeros(len(paths)), np.zeros(len(paths))
		for i in range(len(paths)):
			dis = 1
			for t in range(paths[i]['rew'].shape[0]):
				raw_scores[i] += dis * paths[i]['rew'][t]
				dis *= self.vars['gamma']
			if terminal:
				scores[i] = dis * np.max(terminal.get_preds_np(paths[i]['obs'][-1]))
		scores += raw_scores
		return scores, raw_scores

	def get_rollouts(self, prior=None, num_rollouts=None):
		if prior is None:
			prior = self.planned_actions
		if num_rollouts is None:
			num_rollouts = self.vars['num_rollouts']
		
		if 'Hopper' in self.problem_params['env_type']:
			target_vel = self.env.get_target_vel()
		paths = generate_paths_parallel(copy(self.env), self.env.sim.get_state(),
			prior, self.vars, num_rollouts, target_vel=target_vel)
		return paths

	def advance_plan(self):
		action = self.planned_actions[0]
		self.planned_actions[:-1] = self.planned_actions[1:]
		self.planned_actions[-1] = self.planned_actions[-2]
		return action

def do_rollouts(start_env, start_state, actions, target_vel=None):
	paths = []
	H = actions[0].shape[0]
	num_rollouts = len(actions)

	env = start_env
	if target_vel is not None:
		env.set_target_vel(target_vel)
	for i in range(num_rollouts):
		# env = copy(start_env)
		env.sim.set_state(start_state)

		obss, acts, rews, dones = [], [], [], []

		for k in range(H):
			acts.append(actions[i][k])
			obs, rew, done, _ = env.step(acts[-1])
			obss.append(obs)
			rews.append(rew)
			dones.append(1 if done else 0)

		path = dict(
			obs=np.array(obss), act=np.array(acts), rew=np.array(rews),
			done=np.array(dones)
		)
		paths.append(path)

	return paths

def generate_actions(prior, vars):
	sigma, beta_0, beta_1, beta_2 = vars['filter_coefs']
	eps = np.random.normal(loc=0, scale=1, size=prior.shape) * sigma
	for i in range(2, eps.shape[0]):
		eps[i] = beta_0*eps[i] + beta_1*eps[i-1] + beta_2*eps[i-2]
	return np.clip(prior + eps, vars['min_act'], vars['max_act'])

def generate_paths(start_env, start_state, num_rollouts, prior, vars, target_vel=None):
	actions = []
	for i in range(num_rollouts):
		actions.append(generate_actions(prior, vars))
	return do_rollouts(start_env, start_state, actions, target_vel=target_vel)

def generate_paths_star(args_list):
	return generate_paths(*args_list)

def generate_paths_parallel(start_env, start_state, prior, vars, num_rollouts=None, target_vel=None):
	if num_rollouts is None:
		num_rollouts = vars['num_rollouts']
	rollouts_per_cpu = max(num_rollouts // num_cpu, 1)
	if rollouts_per_cpu * num_cpu != num_rollouts:
		print('WARNING: number of rollouts not divisible by number of CPUs')

	args_list = [start_env, start_state, rollouts_per_cpu, prior, vars, target_vel]
	results = _try_multiprocess(args_list, max_process_time=300, max_timeouts=4)

	paths = []
	for result in results:
		for path in result:
			paths.append(path)

	return paths

def _try_multiprocess(args_list, max_process_time, max_timeouts):
	if max_timeouts == 0:
		print('WARNING: hit maximum number of timeouts in multiprocess')
		return None

	if num_cpu == 1:
		return [generate_paths_star(args_list)]
	else:
		pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
		pruns = [pool.apply_async(generate_paths_star,
			args=(args_list,)) for i in range(num_cpu)]
		try:
			results = [p.get(timeout=max_process_time) for p in pruns]
		except Exception as e:
			print(str(e))
			print('WARNING: timeout error raised, trying again')
			pool.close()
			pool.terminate()
			pool.join()
			return _try_multiprocess(args_list, max_process_time, max_timeouts-1)

		pool.close()
		pool.terminate()
		pool.join()

	return results
