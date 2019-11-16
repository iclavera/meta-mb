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

class MLP(nn.Module):

	def __init__(self, input_size, hidden_sizes, output_size, activation, p):
		super(MLP, self).__init__()

		self.layers = []
		self.activation = activation

		self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
		for i in range(1, len(hidden_sizes)):
			self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
		self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
		
		self.layers = nn.ModuleList(self.layers)

		self.dropout = nn.Dropout(p=p)

		for layer in self.layers:
			nn.init.kaiming_normal_(layer.weight)
			# nn.init.constant_(layer.bias, 0)
			nn.init.normal_(layer.bias, 0, 0.05)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
			if layer != self.layers[-1]:
				x = self.activation(x)
				x = self.dropout(x)

		return x

class ValEnsemble():

	def __init__(self, ensemble_size, input_size, hidden_sizes, output_size, 
				 alpha, activation, kappa, rpf_noise, lam_reg, prior_scale, p):
		self.vals = [MLP(input_size, hidden_sizes, output_size, activation, p) for _ in range(ensemble_size)]
		self.priors = {val: MLP(input_size, hidden_sizes, output_size, activation, p) for val in self.vals}
		self.optims = {
			val: optim.Adam(val.parameters(), lr=alpha, weight_decay=(rpf_noise**2/lam_reg)) for val in self.vals
		}

		self.alpha = alpha
		self.kappa = kappa
		self.rpf_noise = rpf_noise
		self.N = input_size
		self.prior_scale = prior_scale

		for val in self.vals:
			val = val.to(device=device)
		for val in self.priors:
			self.priors[val] = self.priors[val].to(device=device)
			self.priors[val].eval()

	def train_mode(self):
		for val in self.vals:
			val.train()

	def eval_mode(self):
		for val in self.vals:
			val.eval()

	def get_forward(self, val, states):
		pred = val.forward(states) + self.prior_scale * self.priors[val].forward(states)
		return pred

	def get_value(self, state):
		state = torch.tensor(state, dtype=dtype).to(device=device)
		preds = [self.get_forward(val, state) for val in self.vals]
		# print(preds)
		preds = torch.tensor(preds, dtype=dtype)
		# print(preds)
		V_hat = torch.max(preds)
		# print(V_hat)
		return V_hat

	def get_preds(self, state):
		state = torch.tensor(state, dtype=dtype).to(device=device)
		preds = [self.get_forward(val, state) for val in self.vals]
		return preds

	def get_preds_np(self, state):
		state = torch.tensor(state, dtype=dtype).to(device=device)
		preds = np.array([self.get_forward(val, state).detach().cpu() for val in self.vals])
		return preds

	def get_var(self, state):
		state = torch.tensor(state, dtype=dtype).to(device=device)
		preds = np.array([self.get_forward(val, state).detach().cpu() for val in self.vals])
		return np.std(preds)

	def update_val(self, val, buf_states, buf_targets, num_steps, batch_size):
		# Train on the error y_tilde - beta*f_theta_tilde - f_theta
		optim = self.optims[val]
		for _ in range(num_steps):
			# Generate minibatch, and the noisy y_tilde
			inds = np.random.randint(0, buf_states.shape[0], size=batch_size)
			states, targets = buf_states[inds], buf_targets[inds]
			targets += np.random.normal(0, self.rpf_noise, targets.shape)
			states, targets = torch.tensor(states, dtype=dtype), torch.tensor(targets, dtype=dtype)
			states, targets = states.to(device=device), targets.to(device=device)

			preds = self.get_forward(val, states)[:,0]
			# loss = F.smooth_l1_loss(preds, targets) # Huber loss
			loss = F.mse_loss(preds, targets)

			optim.zero_grad()
			loss.backward()
			optim.step()

		return loss

	def update_val_batch(self, val, states, targets, num_steps):
		optim = self.optims[val]

		targets += np.random.normal(0, self.rpf_noise, targets.shape)
		states, targets = torch.tensor(states, dtype=dtype), torch.tensor(targets, dtype=dtype)
		states, targets = states.to(device=device), targets.to(device=device)

		for _ in range(num_steps):
			preds = self.get_forward(val, states)[:,0]
			loss = F.mse_loss(preds, targets)

			optim.zero_grad()
			loss.backward()
			optim.step()

		return loss

	def update_vals_batches(self, buf, num_steps, batch_size, H, gamma):
		size = min(buf.size, buf.total_in)
		inds = np.random.randint(0, size, size=batch_size*len(self.vals))
		states = buf.buffer['state'][inds]
		targets = np.zeros(len(inds))
		comp = {}

		self.eval_mode()
		for i in range(len(inds)):
			if inds[i] in comp:
				targets[i] = comp[inds[i]]
				continue
			dis, max_k = 1, min(H, buf.total_in-inds[i])
			for k in range(max_k):
				targets[i] += dis * buf.get('reward', inds[i]+k)
				dis *= gamma
				if buf.get('done', inds[i]+k):
					break
			targets[i] += dis * np.min(self.get_preds_np(buf.get('next_state', inds[i]+k)))
			comp[inds[i]] = targets[i]

		self.train_mode()
		for i in range(len(self.vals)):
			bi, ei = i*batch_size, (i+1)*batch_size
			self.update_val_batch(self.vals[i], states[bi:ei], targets[bi:ei], num_steps)

		self.eval_mode()

	def update(self, buf_states, buf_targets, num_steps, batch_size):
		avg_loss = 0
		for val in self.vals:
			loss = self.update_val(val, buf_states, buf_targets, num_steps, batch_size)

			# Maintain the last iteration error for each of the networks
			avg_loss += loss.item() / len(self.vals)

		return avg_loss

class PolEnsemble(ValEnsemble):

	def __init__(self, ensemble_size, input_size, hidden_sizes, output_size, 
				 alpha, activation, kappa, rpf_noise, lam_reg, prior_scale, mag_rew, p):
		super(PolEnsemble, self).__init__(ensemble_size, input_size, hidden_sizes, output_size, 
				 						  alpha, activation, kappa, rpf_noise, lam_reg, prior_scale, p)
		self.mag_rew = mag_rew
		self.M = output_size

	def get_action(self, state, mode='random'):
		state = torch.tensor(state, dtype=dtype).to(device=device)
		
		if mode == 'mean':
			preds = [self.get_forward(val, state) for val in self.vals]
			preds = torch.tensor(preds, dtype=dtype)
			act = torch.mean(preds, dim=0)
		elif mode == 'random':
			ind = np.random.randint(0, len(self.vals))
			act = self.get_forward(self.vals[ind], state)

		return act

	def get_forward(self, val, states):
		pred = val.forward(states) + self.prior_scale * torch.tanh(self.priors[val].forward(states))
		# pred = torch.tanh(pred)
		return pred

	def update(self, buf_states, buf_actions, buf_weights, num_steps, batch_size):
		avg_loss = 0
		for val in self.vals:
			loss = self.update_pol(val, buf_states, buf_actions, buf_weights, num_steps, batch_size)
			avg_loss += loss.item() / len(self.vals)
		return avg_loss

	def get_pred(self, ind, state):
		state = torch.tensor(state, dtype=dtype).to(device)
		return self.get_forward(self.vals[ind], state)

	def update_pol(self, val, buf_states, buf_actions, buf_weights, num_steps, batch_size):
		optim = self.optims[val]
		for _ in range(num_steps):
			# Generate minibatch, and the noisy y_tilde
			inds = np.random.randint(0, buf_states.shape[0], size=batch_size)
			states, actions, weights = buf_states[inds], buf_actions[inds], buf_weights[inds]
			actions += np.random.normal(0, self.rpf_noise, actions.shape)
			states, actions, weights = torch.tensor(states, dtype=dtype), torch.tensor(actions, dtype=dtype), torch.tensor(weights, dtype=dtype)
			states, actions, weights = states.to(device=device), actions.to(device=device), weights.to(device=device)

			preds, loss = self.get_forward(val, states), 0
			for i in range(batch_size):
				loss += weights[i] * torch.norm(preds[i] - actions[i], p=2)
				loss -= self.mag_rew * torch.norm(preds[i], p=2)
			loss /= batch_size
			
			optim.zero_grad()
			loss.backward()
			optim.step()

		return loss

	def update_pol_batch(self, val, states, targets, weights, num_steps):
		optim = self.optims[val]
		targets += np.random.normal(0, self.rpf_noise, targets.shape)
		states, targets = torch.tensor(states, dtype=dtype), torch.tensor(targets, dtype=dtype)
		states, targets = states.to(device=device), targets.to(device=device)

		for _ in range(num_steps):
			preds, loss = self.get_forward(val, states), 0
			for i in range(states.shape[0]):
				loss += weights[i] * torch.norm(preds[i]-targets[i], p=2)
			loss /= states.shape[0]

			optim.zero_grad()
			loss.backward()
			optim.step()

	def update_pols_batches(self, buf, val_ens, num_steps, batch_size, gamma, exp_beta):
		size = min(buf.size, buf.total_in)
		inds = np.random.randint(0, size, size=batch_size*len(self.vals))
		states = buf.buffer['state'][inds]
		targets = np.zeros((len(inds), self.M))
		weights = np.ones(len(inds))
		comp = {}

		self.eval_mode()
		for i in range(len(inds)):
			targets[i] = buf.get('action', inds[i])
		# 	if inds[i] in comp:
		# 		weights[i] = comp[inds[i]]
		# 		continue
		# 	weights[i] += gamma * val_ens.get_value(buf.get('next_state', inds[i])).item()
		# 	weights[i] += buf.get('reward', inds[i])
		# 	weights[i] -= val_ens.get_value(buf.get('state', inds[i])).item()
		# 	comp[inds[i]] = weights[i]

		# advs_std, advs_avg = np.std(weights), np.mean(weights)
		# weights = (weights - advs_avg) / (advs_std + 1e-6)
		# weights = np.exp(exp_beta * weights)

		self.train_mode()
		if True:
			for i in range(len(self.vals)):
				bi, ei = i*batch_size, (i+1)*batch_size
				self.update_pol_batch(self.vals[i], states[bi:ei], targets[bi:ei], weights[bi:ei], num_steps)
		else:
			# some kind of CUDA error is going on
			pool = mp.Pool(processes=len(self.vals), maxtasksperchild=1)
			pruns = []
			for i in range(len(self.vals)):
				bi, ei = i*batch_size, (i+1)*batch_size
				pruns.append(
					pool.apply_async(self.update_pol_batch,
						args=(self.vals[i], states[bi:ei], targets[bi:ei], weights[bi:ei], num_steps)))
			for p in pruns:
				p.get(timeout=3600*24*100)
			pool.close()
			pool.terminate()
			pool.join()

		self.eval_mode()
