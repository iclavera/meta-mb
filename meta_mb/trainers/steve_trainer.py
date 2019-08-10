import tensorflow as tf
import numpy as np
import time
from meta_mb.logger import logger

import abc
from collections import OrderedDict
from distutils.version import LooseVersion
from itertools import count
import math
import os
from pdb import set_trace as st
from meta_mb.replay_buffers import SimpleReplayBuffer

class Trainer(object):
    """
    Performs steps for MAML
    Args:
        algo (Algo) :
        env (Env) :
        env_sampler (Sampler) :
        env_sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            algo,
            env,
            ):
        self.algo = algo
        self.env = env

        if sess is None:
            sess = tf.Session()
        self.sess = sess


    def train(self):
        """
        Trains policy on env using algo
        Pseudocode:
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """

        with self.sess.as_default() as sess:
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))
            start_time = time.time()
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=10, pad_step_number=True)

            # if self.start_itr == 0:
            if tf.train.get_checkpoint_state(self.restore_path):
                logger.log("reading stored model from %s", self.restore_path)
                self.saver.restore(sess, tf.train.latest_checkpoint(self.restore_path))
                self.start_itr = self.itr.eval() + 1

            else:
                self.start_itr = 0
                self.algo._update_target(tau=1.0)
                while self.env_replay_buffer._size < self.n_initial_exploration_steps:
                    paths = self.env_sampler.obtain_samples(log=True, log_prefix='train-', random=True)
                    samples_data = self.env_sample_processor.process_samples(paths, log='all', log_prefix='train-')
                    self.env_replay_buffer.add_samples(samples_data['observations'], samples_data['actions'], samples_data['rewards'],
                                                       samples_data['dones'], samples_data['next_observations'])


            for itr in range(self.start_itr, self.n_itr):
                self.itr = itr
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                paths = self.env_sampler.obtain_samples(log=True, log_prefix='train-')
                samples_data = self.env_sample_processor.process_samples(paths, log='all', log_prefix='train-')

                paths = self.env_sampler.obtain_samples(log=True, log_prefix='eval-', deterministic=True)
                _ = self.env_sample_processor.process_samples(paths, log='all', log_prefix='eval-')

                self.log_diagnostics(paths, prefix='train-')

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('rollout_length', self.rollout_length)
                logger.logkv('n_timesteps', self.env_sampler.total_timesteps_sampled)
                logger.logkv('ItrTime', time.time() - itr_start_time)
                logger.logkv('SAC Training Time', sum(sac_time))
                logger.logkv('Model Rollout Time', sum(expand_model_replay_buffer_time))

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()
                if itr == 0:
                    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()


    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
