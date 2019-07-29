from meta_mb.logger import logger
import matplotlib.pyplot as plt
import numpy as np
from meta_mb.optimizers.base import Optimizer
from meta_mb.utils import Serializable
import tensorflow as tf
import os


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
