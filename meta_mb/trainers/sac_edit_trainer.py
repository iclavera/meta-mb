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
from random import randint



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
            env_sampler,
            env_sample_processor,
            model_sample_processor,
            dynamics_model,
            policy,
            n_itr,
            rollout_length_params,
            dynamics_model_max_epochs=200,
            policy_update_per_iteration=20,
            num_model_rollouts = 1,
            start_itr=0,
            task=None,
            sess=None,
            n_initial_exploration_steps=1e3,
            env_num_grad_steps=None,
            model_num_grad_steps=None,
            env_max_replay_buffer_size=1e6,
            model_max_replay_buffer_size=4*1e7,
            speed_up_factor=100,
            rollout_batch_size=100,
            model_train_freq=100,
            n_train_repeats=1,
            real_ratio = 1,
            rollout_length = 1
            ):
        self.algo = algo
        self.env = env
        self.env_sampler = env_sampler
        self.env_sample_processor = env_sample_processor
        self.model_sample_processor = model_sample_processor
        self.dynamics_model = dynamics_model
        self.baseline = env_sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_model_rollouts = num_model_rollouts
        self.task = task
        self.rollout_length_params = rollout_length_params
        self.n_initial_exploration_steps = n_initial_exploration_steps
        self.env_replay_buffer = SimpleReplayBuffer(self.env, env_max_replay_buffer_size)
        self.model_replay_buffer = SimpleReplayBuffer(self.env, model_max_replay_buffer_size)
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.env_num_grad_steps = env_num_grad_steps
        self.rollout_batch_size = rollout_batch_size
        self.rollout_length = rollout_length
        self.speed_up_factor = speed_up_factor
        if env_num_grad_steps is None:
            self.env_num_grad_steps = self.env_sampler.total_samples
        # if model_num_grad_steps is None:
        self.model_num_grad_steps = policy_update_per_iteration * self.env_num_grad_steps
        self.policy_update_per_iteration = policy_update_per_iteration
        self.model_train_freq = model_train_freq
        self.n_train_repeats = n_train_repeats
        self.real_ratio = real_ratio
        #     self.model_num_grad_steps = 20 * self.env_sampler.total_samples

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

            if self.start_itr == 0:
                self.algo._update_target(tau=1.0)
                if self.n_initial_exploration_steps > 0:
                    while self.env_replay_buffer._size <= self.n_initial_exploration_steps:
                        paths = self.env_sampler.obtain_samples(log=True, log_prefix='train-', random=True)
                        samples_data = self.env_sample_processor.process_samples(paths, log='all', log_prefix='train-')
                        self.env_replay_buffer.add_samples(samples_data['observations'], samples_data['actions'], samples_data['rewards'],
                                                            samples_data['dones'], samples_data['next_observations'])

            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                """ -------------------- Sampling --------------------------"""
                all_samples = self.env_replay_buffer.all_samples()
                for _ in range(self.env_sampler.max_path_length // self.model_train_freq):
                    self.dynamics_model.fit(all_samples[0], all_samples[1], all_samples[2],
                                            epochs=self.dynamics_model_max_epochs, verbose=False,
                                            log_tabular=True, prefix='Model-')
                    for _ in range(self.rollout_length):
                        random_states = self.env_replay_buffer.random_batch_simple(int(self.rollout_batch_size))['observations']
                        actions_from_policy = self.policy.get_actions(random_states)[0]
                        predictions = self.dynamics_model.predict(random_states, actions_from_policy)
                        rewards = self.algo.training_environment.reward(random_states, actions_from_policy, predictions)
                        terminals = np.array([False]).repeat(random_states.shape[0])
                        self.model_replay_buffer.add_samples(random_states, actions_from_policy, rewards, terminals, predictions)
                    self.set_rollout_length(itr)

                logger.log("Obtaining samples...")
                time_env_sampling_start = time.time()
                paths = self.env_sampler.obtain_samples(log=True, log_prefix='train-')
                sampling_time = time.time() - time_env_sampling_start

                """ ----------------- Processing Samples ---------------------"""
                # check how the samples are processed
                logger.log("Processing samples...")
                time_proc_samples_start = time.time()
                samples_data = self.env_sample_processor.process_samples(paths, log='all', log_prefix='train-')
                sample_num = samples_data['observations'].shape[0]
                self.env_replay_buffer.add_samples(samples_data['observations'], samples_data['actions'], samples_data['rewards'],
                                                    samples_data['dones'], samples_data['next_observations'])
                proc_samples_time = time.time() - time_proc_samples_start
                model_start = time.time()
                paths = self.env_sampler.obtain_samples(log=True, log_prefix='eval-', deterministic=True)
                _ = self.env_sample_processor.process_samples(paths, log='all', log_prefix='eval-')

                for _ in range(self.n_train_repeats * self.env_sampler.max_path_length):
                    batch_size = 256
                    env_batch_size = int(batch_size*self.real_ratio)
                    model_batch_size = batch_size - env_batch_size
                    env_batch = self.env_replay_buffer.random_batch(env_batch_size)
                    model_batch = self.model_replay_buffer.random_batch(int(model_batch_size))
                    keys = env_batch.keys()
                    batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
                    self.algo.do_training(itr, batch)


                self.log_diagnostics(paths, prefix='train-')


                train_actor_start = time.time()

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.env_sampler.total_timesteps_sampled)
                # logger.logkv('fit_model_time', model_end - model_start)
                # logger.logkv('train_critic_time', critic_end - time_optimization_step_start)
                # logger.logkv('train_actor_time', sum(actor_time))
                # logger.logkv('predict_time', train_actor_start - predict_start)
#
                # logger.logkv('Time-Optimization', time.time() - time_optimization_step_start)
                # logger.logkv('Time-SampleProc', np.sum(proc_samples_time))
                # logger.logkv('Time-Sampling', sampling_time)

                # logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()
                if itr == 0:
                    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()

    def set_rollout_length(self, itr):
        min_epoch, max_epoch, min_length, max_length = self.rollout_length_params
        if itr <= min_epoch:
            y = min_length
        else:
            dx = (itr - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self.rollout_length = int(y)
        # print('[ Model Length ] Epoch: {} (min: {}, max: {}) | Length: {} (min: {} , max: {})'.format(
        #     itr, min_epoch, max_epoch, self._rollout_length, min_length, max_length
        # ))

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

        # feed_dict = self._get_feed_dict(iteration, batch)
        # # TODO(hartikainen): We need to unwrap self._diagnostics_ops from its
        # # tensorflow `_DictWrapper`.
        # diagnostics = self._session.run({**self._diagnostics_ops}, feed_dict)
        #
        # diagnostics.update(OrderedDict([
        #     (f'policy/{key}', value)
        #     for key, value in
        #     self._policy.get_diagnostics(flatten_input_structure({
        #         name: batch['observations'][name]
        #         for name in self._policy.observation_keys
        #     })).items()
        # ]))
        #
        # if self._plotter:
        #     self._plotter.draw()
        #
        # return diagnostics
        #
        # return diagnostics
