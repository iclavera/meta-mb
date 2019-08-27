import tensorflow as tf
import numpy as np
import time
from meta_mb.logger import logger
# import nn

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
            env_sampler,
            env_sample_processor,
            dynamics_model,
            done_predictor,
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
            done_bar=1.0,
            dynamics_type=0,
            step_type=0,
            aux_hidden_dim=256,
            predict_done=False,
            actorH=1,
            T=1,
            ground_truth=False,
            ):
        self.algo = algo
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
        self.epoch_length = self.env_sampler.max_path_length - 1
        self.model_train_freq = model_train_freq
        self.restore_path = restore_path
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.sampler_batch_size = sampler_batch_size
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.action_dim = int(np.prod(self.env.action_space.shape))
        self.done_bar = done_bar
        self.dynamics_type = dynamics_type
        self.step_type = step_type
        self.done_predictor = done_predictor
        self.predict_done = predict_done
        self.actorH = actorH
        self.T = T
        self.ground_truth = ground_truth

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

            if self.algo.obs_dim > 50:
                self.deal_with_oom = 16
            else:
                self.deal_with_oom = 1

            time_step = 0
            for itr in range(self.start_itr, self.n_itr):
                self.itr = itr
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")
                paths = self.env_sampler.obtain_samples(log=True, log_prefix='train-')
                samples_data = self.env_sample_processor.process_samples(paths, log='all', log_prefix='train-')
                fit_start = time.time()
                all_samples = self.env_replay_buffer.all_samples()
                logger.log("Training models...")
                if self.dynamics_type == 0 or self.dynamics_type == 3:
                    self.dynamics_model.fit(all_samples[0], all_samples[1], all_samples[2],
                                            epochs=self.dynamics_model_max_epochs, verbose=False,
                                            log_tabular=True, prefix='Model-')
                elif self.dynamics_type == 1 or self.dynamics_type == 2:
                    expanded_rewards = np.expand_dims(all_samples[3], axis = -1)
                    self.dynamics_model.fit(all_samples[0], all_samples[1], np.concatenate([all_samples[2], expanded_rewards], axis = -1),
                                            epochs=self.dynamics_model_max_epochs, verbose=False,
                                            log_tabular=True, prefix='Model-')
                if self.predict_done:
                    input = np.concatenate([all_samples[0], all_samples[1], all_samples[2]], -1)
                    output = all_samples[4]
                    self.done_predictor.fit(input, output)
                logger.logkv('Fit model time', time.time() - fit_start)
                logger.log("Done training models...")
                expand_model_replay_buffer_time = []
                sac_time = []

                # for h in range(self.actorH):
                for _ in range(self.epoch_length // self.model_train_freq):
                    expand_model_replay_buffer_start = time.time()
                    for _ in range(self.rollout_length):
                        for _ in range(self.deal_with_oom):
                            samples_num = int(self.rollout_batch_size)//(self.deal_with_oom)
                            random_states = self.env_replay_buffer.random_batch_simple(samples_num)['observations']
                            actions_from_policy = self.policy.get_actions(random_states)[0]
                            next_obs, rewards, term = self.step(random_states, actions_from_policy)
                            self.model_replay_buffer.add_samples(random_states,
                                                                 actions_from_policy,
                                                                 rewards,
                                                                 term,
                                                                 next_obs)
                    self.set_rollout_length(itr)
                    expand_model_replay_buffer_time.append(time.time() - expand_model_replay_buffer_start)

                    sac_start = time.time()
                    for _ in range(self.model_train_freq):
                        for _ in range(self.n_train_repeats):
                            batch_size = self.sampler_batch_size
                            env_batch_size = int(batch_size * self.real_ratio)
                            model_batch_size = batch_size - env_batch_size
                            env_batch = self.env_replay_buffer.random_batch(env_batch_size)
                            model_batch = self.model_replay_buffer.random_batch(int(model_batch_size))
                            keys = env_batch.keys()
                            batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
                            reward = []
                            for t in range(self.T):
                                next_obs = batch['next_observations']
                                next_actions = self.policy.get_actions(next_obs)[0]
                                result = self.env.step_batch(next_obs, next_actions, t, reward)
                            batch.update(result)
                            self.algo.do_training(time_step, batch, log=True)
                    sac_time.append(time.time() - sac_start)
                self.env_replay_buffer.add_samples(samples_data['observations'],
                                                   samples_data['actions'],
                                                   samples_data['rewards'],
                                                   samples_data['dones'],
                                                   samples_data['next_observations'])
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
                # if itr == 0:
                #     sess.graph.finalize()

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

    def step(self, obs, actions):
        assert self.dynamics_type in [0, 1, 2, 3]
        assert self.step_type in [0, 1]
        if self.dynamics_type == 0 or self.dynamics_type == 3:
            next_observation = self.dynamics_model.predict(obs, actions)
            rewards = self.env.reward(obs, actions, next_observation)
            if self.predict_done:
                dones = self.done_predictor.predict(obs, actions, next_observation)
            else:
                dones = self.env.termination_fn(obs, actions, next_observation).reshape((-1))
        elif self.dynamics_type == 1 or self.dynamics_type == 2:
            next_observation, rewards = self.dynamics_model.predict(obs, actions)
            if self.predict_done:
                dones = self.done_predictor.predict(obs, actions, next_observation)
            else:
                dones = self.env.termination_fn(obs, actions, next_observation).reshape((-1))
            if self.step_type == 1:
                rewards = self.env.reward(obs, actions, next_observation)
        dones = dones.reshape((-1))
        rewards = rewards.reshape((-1))
        return next_observation, rewards, dones

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
