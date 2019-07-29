from meta_mb.samplers.base import BaseSampler
from meta_mb.utils.serializable import Serializable
from meta_mb.samplers.vectorized_env_executor import ParallelEnvExecutor, IterativeEnvExecutor
from meta_mb.logger import logger
from meta_mb.utils import utils

from pyprind import ProgBar
import numpy as np
import time
import os
import itertools
import matplotlib.pyplot as plt
from math import ceil
from itertools import accumulate


class GTSampler(BaseSampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_mb.meta_envs.base.MetaEnv) : environment object
        policy (meta_mb.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of meta_envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(
            self,
            env,
            policy,
            num_rollouts,
            max_path_length,
            dyn_pred_str=None,
            vae=None,
    ):
        Serializable.quick_init(self, locals())
        super(GTSampler, self).__init__(env, policy, num_rollouts, max_path_length)

        self.total_samples = num_rollouts * max_path_length
        self.total_timesteps_sampled = 0
        self.vae = vae
        self.dyn_pred_str = dyn_pred_str

    def update_tasks(self):
        pass

    def obtain_samples(self, log, log_prefix='', deterministic=False, verbose=True, plot_first_rollout=False):
        self.policy.reset()  # do not reset
        returns_array = self.policy.get_rollouts(
            deterministic=deterministic, plot_first_rollout=plot_first_rollout
        )

        logger.logkv(log_prefix + 'AverageReturn', np.mean(returns_array))
        if log:
            for idx, returns in enumerate(returns_array):
                logger.logkv(log_prefix + f'Return {idx}', returns)
            # logger.logkv(log_prefix + 'StdReturn', np.std(returns_array))
            # logger.logkv(log_prefix + 'MaxReturn', np.max(returns_array))
            # logger.logkv(log_prefix + 'MinReturn', np.min(returns_array))

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        # dumps policy
        state['policy'] = self.policy.__getstate__()
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        self.policy = state['policy']
