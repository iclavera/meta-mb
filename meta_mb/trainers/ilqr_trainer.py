import tensorflow as tf
import time
from meta_mb.logger import logger


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
            algo,
            env,
            env_sampler,
            dynamics_sample_processor,
            policy,
            dynamics_model,
            n_itr,
            steps_per_iter,
            initial_random_samples=True,
            sess=None,
            dynamics_model_max_epochs=200,
            ):
        self.algo = algo
        self.env = env
        self.env_sampler = env_sampler
        self.dynamics_sample_processor = dynamics_sample_processor
        self.dynamics_model = dynamics_model
        self.policy = policy
        self.n_itr = n_itr
        self.dynamics_model_max_epochs = dynamics_model_max_epochs

        self.initial_random_samples = initial_random_samples
        self.steps_per_iter = steps_per_iter

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
            # uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            # sess.run(tf.variables_initializer(uninit_vars))
            sess.run(tf.global_variables_initializer())

            start_time = time.time()
            for itr in range(self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                time_env_sampling_start = time.time()

                if self.initial_random_samples and itr == 0:
                    logger.log("Obtaining random samples from the environment...")
                    env_paths = self.env_sampler.obtain_samples(log=True, random=True, log_prefix='')

                else:
                    logger.log("Obtaining samples from the environment using the policy...")
                    env_paths = self.env_sampler.obtain_samples(log=True, log_prefix='')

                # Add sleeping time to match parallel experiment
                # time.sleep(10)

                logger.record_tabular('Data-TimeEnvSampling', time.time() - time_env_sampling_start)
                logger.log("Processing environment samples...")

                # first processing just for logging purposes
                time_env_samp_proc = time.time()

                samples_data = self.dynamics_sample_processor.process_samples(env_paths, log=True,
                                                                              log_prefix='Data-EnvTrajs-')

                self.env.log_diagnostics(env_paths, prefix='Data-EnvTrajs-')

                logger.record_tabular('Data-TimeEnvSampleProc', time.time() - time_env_samp_proc)

                ''' --------------- fit dynamics model --------------- '''

                time_fit_start = time.time()

                logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
                self.dynamics_model.fit(samples_data['observations'],
                                        samples_data['actions'],
                                        samples_data['next_observations'],
                                        epochs=self.dynamics_model_max_epochs, verbose=False,
                                        log_tabular=True, prefix='Model-')

                logger.record_tabular('Model-TimeModelFit', time.time() - time_fit_start)

                ''' --------------- policy updates --------------- '''
                time_opt_start = time.time()

                for step in range(self.steps_per_iter):

                    logger.log("\n ---------------- Grad-Step %d ----------------" % step)

                    logger.log("Optimizing policy...")
                    # This needs to take all samples_data so that it can construct graph for meta-optimization.
                    self.algo.optimize_policy(samples_data)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Iteration', itr)
                logger.logkv('n_timesteps', self.env_sampler.total_timesteps_sampled)
                logger.logkv('Policy-TimeStep', time.time() - time_opt_start)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.dumpkvs()
                if itr == 0:
                    sess.graph.finalize()

            logger.logkv('Trainer-TimeTotal', time.time() - start_time)

        logger.log("Training finished")
        self.sess.close()
