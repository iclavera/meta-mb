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
            env_name,
            env_sampler,
            env_sample_processor,
            dynamics_model,
            policy,
            n_itr,
            rollout_length_params,
            num_model_rollouts = 1,
            sess=None,
            n_initial_exploration_steps=1e3,
            env_max_replay_buffer_size=1e6,
            model_max_replay_buffer_size=2e6,
            rollout_batch_size=100,
            n_train_repeats=1,
            real_ratio = 1,
            rollout_length = 1,
            model_deterministic = False,
            model_train_freq=250,
            restore_path=None,
            dynamics_model_max_epochs=50,
            sampler_batch_size=64,
            ):
        self.algo = algo
        self.env_name = env_name
        self.env = env
        self.env_sampler = env_sampler
        self.env_sample_processor = env_sample_processor
        self.dynamics_model = dynamics_model
        self.baseline = env_sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.num_model_rollouts = num_model_rollouts
        self.rollout_length_params = rollout_length_params
        self.n_initial_exploration_steps = n_initial_exploration_steps
        self.env_replay_buffer = SimpleReplayBuffer(self.env, env_max_replay_buffer_size)
        self.model_replay_buffer = SimpleReplayBuffer(self.env, model_max_replay_buffer_size)
        self.rollout_batch_size = rollout_batch_size
        self.rollout_length = rollout_length
        self.n_train_repeats = n_train_repeats
        self.real_ratio = real_ratio
        self.model_deterministic = model_deterministic
        self.model_train_freq = model_train_freq
        self.restore_path = restore_path
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.sampler_batch_size = sampler_batch_size
        self.epoch_length = self.env_sampler.max_path_length - 1


        if sess is None:
            sess = tf.Session()
        self.sess = sess


    def termination_fn(self, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        if self.env_name == 'AntEnv':
            x = next_obs[:, 0]
            not_done = 	np.isfinite(next_obs).all(axis=-1) * (x >= 0.2) * (x <= 1.0)
            done = ~not_done
        elif self.env_name == 'HalfCheetahEnv' or 'Walker2dEnv':
            done = np.array([False]).repeat(obs.shape[0])
        elif self.env_name == 'HopperEnv':
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done =  np.isfinite(next_obs).all(axis=-1) \
                        * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                        * (height > .7) \
                        * (np.abs(angle) < .2)

            done = ~not_done
        else:
            print("Error")
            return
        done = done[:,None]
        return done

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

            self.start_itr = 0
            self.algo._update_target(tau=1.0)
            while self.env_replay_buffer._size < self.n_initial_exploration_steps:
                paths = self.env_sampler.obtain_samples(log=True, log_prefix='train-', random=True)
                samples_data = self.env_sample_processor.process_samples(paths, log='all', log_prefix='train-')
                self.env_replay_buffer.add_samples(samples_data['observations'], samples_data['actions'], samples_data['rewards'],
                                                    samples_data['dones'], samples_data['next_observations'])


            for itr in range(self.start_itr, self.n_itr):
                self.itr = itr
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")
                time_step = 0

                paths = self.env_sampler.obtain_samples(log=True, log_prefix='train-')
                samples_data = self.env_sample_processor.process_samples(paths, log='all', log_prefix='train-')
                fit_start = time.time()
                all_samples = self.env_replay_buffer.all_samples()
                logger.log("Training models...")
                # obs, actions, nect_obs, rewards, dones
                self.dynamics_model.fit(all_samples[0], all_samples[1], all_samples[2], all_samples[3], all_samples[4],
                                            epochs=self.dynamics_model_max_epochs, verbose=False,
                                            log_tabular=True, prefix='Model-')
                logger.logkv('Fit model time', time.time() - fit_start)
                logger.log("Done training models...")
                expand_model_replay_buffer_time = []
                sac_time = []
                for _ in range(self.epoch_length // self.model_train_freq):
                    expand_model_replay_buffer_start = time.time()
                    for _ in range(self.rollout_length):
                        random_states = self.env_replay_buffer.random_batch_simple(int(self.rollout_batch_size))['observations']
                        actions_from_policy = self.policy.get_actions(random_states)[0]
                        next_obs = self.dynamics_model.predict(random_states, actions_from_policy)
                        term = self.termination_fn(random_states, actions_from_policy, next_obs)
                        term = term.reshape((-1))
                        rewards = self.env.reward(random_states, actions_from_policy, next_obs)
                        self.model_replay_buffer.add_samples(random_states, actions_from_policy, rewards, term, next_obs)
                    self.set_rollout_length(itr)
                    expand_model_replay_buffer_time.append(time.time() - expand_model_replay_buffer_start)

                    sac_start = time.time()
                    for _ in range(self.model_train_freq):
                        time_step += 1
                        for _ in range(self.n_train_repeats):
                            batch_size = self.sampler_batch_size
                            env_batch_size = int(batch_size*self.real_ratio)
                            model_batch_size = batch_size - env_batch_size
                            env_batch = self.env_replay_buffer.random_batch(env_batch_size)
                            model_batch = self.model_replay_buffer.random_batch(int(model_batch_size))
                            keys = env_batch.keys()
                            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
                            self.algo.do_training(itr * self.epoch_length + time_step, batch)
                    sac_time.append(time.time() - sac_start)
                self.env_replay_buffer.add_samples(samples_data['observations'], samples_data['actions'], samples_data['rewards'],
                                                        samples_data['dones'], samples_data['next_observations'])
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



    def set_rollout_length(self, itr):
        min_epoch, max_epoch, min_length, max_length = self.rollout_length_params
        if itr <= min_epoch:
            y = min_length
        else:
            dx = (itr - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length

        self.rollout_length = int(y)


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
