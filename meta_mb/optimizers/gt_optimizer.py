from meta_mb.logger import logger
import matplotlib.pyplot as plt
import numpy as np
from meta_mb.optimizers.base import Optimizer
from meta_mb.utils import Serializable
import tensorflow as tf
import os
import copy


class GTOptimizer(Optimizer, Serializable):
    def __init__(
            self,
            alpha,
            beta1=0.9,
            beta2=0.999,
            epsilon=10e-8,
    ):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def compute_delta_var(self, grad):
        """

        :param grad_tau: numpy array with shape (horizon, batch_size, act_dims)
        :return:
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1-self.beta1) * grad
        self.v = self.beta2 * self.v + (1-self.beta2) * grad * grad
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)


class LookaheadOptimizer(object):
    """
    MINIMIZE loss
    """
    def __init__(self, alpha, k, grad_fn, inner_optimizer):
        self.alpha = alpha
        self.k = k
        self.grad_fn = grad_fn
        self.inner_optimizer = inner_optimizer

    def compute_delta_var(self, phi):
        theta = phi.copy()
        for i in range(self.k):
            theta += self.inner_optimizer.compute_delta_var(self.grad_fn(theta))
        return phi

class CollocationProblem(object):
    def __init__(self, env, horizon, act_dim, obs_dim, discount, lmbda):
        self._env = copy.deepcopy(env)
        self.horizon = horizon
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.discount = discount
        self.lmbda = lmbda

    def objective(self, s_a_array_stacked):
        s_array, a_array = s_a_array_stacked[:self.horizon], s_a_array_stacked[self.horizon:]
        sum_rewards, reg_loss = 0, 0
        for t in range(self.horizon):
            _ = self._env.reset_from_obs(s_array[t])
            s_next, reward, _, _ = self._env.step(a_array[t])
            sum_rewards += self.discount**t * reward
            if t < self.horizon - 1:
                reg_loss += np.linalg.norm(s_array[t+1] - s_next)**2
        return -sum_rewards + reg_loss

