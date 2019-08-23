import tensorflow as tf
import numpy as np
import time
from meta_mb.logger import logger

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class Trainer(object):
    """
    Performs steps for MAML

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            env,
            sampler,
            dynamics_sample_processor,
            policy,
            dynamics_model,
            n_itr,
            start_itr=0,
            initial_random_samples=True,
            sess=None,
            dynamics_model_max_epochs=200,
            reward_model_max_epochs=200,
            reward_model=None,
            reward_sample_processor=None,
            plot_open_loop= False,
            ):
        self.env = env
        self.sampler = sampler
        self.dynamics_sample_processor = dynamics_sample_processor
        self.dynamics_model = dynamics_model
        self.reward_sample_processor = reward_sample_processor
        self.reward_model = reward_model
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.reward_model_max_epochs = reward_model_max_epochs
        self.plot_open_loop = plot_open_loop

        self.initial_random_samples = initial_random_samples

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
            name_to_var = {v.op.name: v for v in tf.global_variables() + tf.local_variables()}
            uninit_vars_b = sess.run(tf.report_uninitialized_variables())
            uninit_vars = list(name_to_var[name.decode()] for name in uninit_vars_b)
            sess.run(tf.variables_initializer(uninit_vars))

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                time_env_sampling_start = time.time()

                if self.initial_random_samples and itr == 0:
                    logger.log("Obtaining random samples from the environment...")
                    env_paths = self.sampler.obtain_samples(log=True, random=True, log_prefix='')
                else:
                    logger.log("Obtaining samples from the environment using the policy...")
                    env_paths = self.sampler.obtain_samples(log=True, log_prefix='')


                # import pdb; pdb.set_trace()

                logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)
                logger.log("Processing environment samples...")

                # first processing just for logging purposes
                time_env_samp_proc = time.time()
                samples_data = self.dynamics_sample_processor.process_samples(env_paths,
                                                                              log=True, log_prefix='EnvTrajs-')

                logger.record_tabular('Time-EnvSampleProc', time.time() - time_env_samp_proc)

                ''' --------------- fit dynamics model --------------- '''

                time_fit_start = time.time()

                logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
                if self.dynamics_model_max_epochs > 0:
                    self.dynamics_model.fit(samples_data['observations'],
                                            samples_data['actions'],
                                            samples_data['next_observations'],
                                            epochs=self.dynamics_model_max_epochs, verbose=False, log_tabular=True)

                logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)

                ''' --------------- fit reward model --------------- '''
                time_fitrew_start = time.time()

                if self.reward_model is not None:
                    if self.reward_sample_processor is not None:
                        samples_data = self.reward_sample_processor.process_samples(env_paths, log=False)
                    logger.log("Training reward model for %i epochs ..." % (self.reward_model_max_epochs))
                    self.reward_model.fit(samples_data['observations'],
                                            samples_data['actions'],
                                            samples_data['next_observations'],
                                            samples_data['rewards'],
                                            epochs=self.reward_model_max_epochs, verbose=False, log_tabular=True,
                                            prefix='Rew')

                    logger.record_tabular('Time-RewardModelFit', time.time() - time_fitrew_start)

                """ -------------------- Plot open loop rollout -------------"""
                if self.plot_open_loop and itr % 10 == 0:
                    # h = 249
                    env_paths_random = self.sampler.obtain_samples(log=True, random=True, log_prefix='')
                    env_paths_mpc = self.sampler.obtain_samples(log=True, log_prefix='')
                    one_step_error = []
                    for env_paths in [env_paths_random, env_paths_mpc]:
                        actions = np.stack([path['actions'] for path in env_paths])
                        true_states = np.stack([path['observations'] for path in env_paths])

                        states = true_states[:, :-1].reshape((-1, true_states.shape[-1]))
                        actions = actions.reshape((-1, actions.shape[-1]))
                        next_states = true_states[:, 1:].reshape((-1, true_states.shape[-1]))
                        predicted_next_states = sess.run(self.dynamics_model.predict_sym(states.astype(np.float32),
                                                                                         actions.astype(np.float32)))
                        error = next_states - predicted_next_states
                        one_step_error.append(error)

                    logger.logkv('OneStepErrorRandomTraj', np.mean(np.linalg.norm(one_step_error[0], axis=1)))
                    logger.logkv('OneStepErrorMPCTraj', np.mean(np.linalg.norm(one_step_error[1], axis=1)))


                    # openloop_predicted_states = self.dynamics_model.openloop_rollout(true_states[:, 0], actions)
                    #
                    # for counter, (true_traj, predicte_openloop_traj) in \
                    #         enumerate(zip(true_states[:, 1:h + 1], openloop_predicted_states)):
                    #     assert len(self.env.observation_space.shape) == 1
                    #     num_plots = self.env.observation_space.shape[0]
                    #     fig = plt.figure(figsize=(num_plots, 6))
                    #
                    #     for i in range(num_plots):
                    #         ax = plt.subplot(num_plots // 6 + 1, 6, i + 1)
                    #         plt.plot(true_traj[:, i], label='true state')
                    #         plt.plot(predicte_openloop_traj[:, i], label='openloop')
                    #     plt.savefig(os.path.join(logger.get_dir(), 'diag_openloop_itr%d_%d.png' % (itr, counter)))


                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                # logger.log("Saving snapshot...")
                # params = self.get_itr_snapshot(itr)
                self.log_diagnostics(env_paths, '')
                # logger.save_itr_params(itr, params)
                # logger.log("Saved")

                logger.dumpkvs()
                #if itr == 0:
                #    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, dynamics_model=self.dynamics_model, reward_model=self.reward_model)

    def log_diagnostics(self, paths, prefix):
        if hasattr(self.env, 'log_diagnostics'):
            self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
