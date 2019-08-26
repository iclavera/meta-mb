import tensorflow as tf
import time
from meta_mb.logger import logger


class BPTTTrainer(object):
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
            initial_random_samples=True,
            initial_sinusoid_samples=False,
            sess=None,
            dynamics_model_max_epochs=200,
            fit_model=True,
            on_policy_freq=1,
            cem_sampler=None,
            use_pretrained_model=False,
            num_random_iters=1,
    ):
        self.env = env
        self.sampler = sampler
        self.dynamics_sample_processor = dynamics_sample_processor
        self.dynamics_model = dynamics_model
        self.policy = policy
        self.n_itr = n_itr
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.fit_model = fit_model
        self.use_pretrained_model = use_pretrained_model
        if on_policy_freq is not None:
            assert cem_sampler is not None
        self.on_policy_freq = on_policy_freq
        self.cem_sampler = cem_sampler

        self.initial_random_samples = initial_random_samples and not use_pretrained_model
        self.initial_sinusoid_samples = initial_sinusoid_samples and not use_pretrained_model
        self.num_random_iters = num_random_iters

        if sess is None:
            sess = tf.Session()
        self.sess = sess


    def train(self):
        logger.log('training starts...')
        with self.sess.as_default() as sess:
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            if self.dynamics_model is not None:
                if self.use_pretrained_model:
                    global_vars = tf.global_variables()
                    is_var_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
                    uninit_vars = [var for idx, var in enumerate(global_vars) if not is_var_initialized[idx]]
                    sess.run(tf.variables_initializer(uninit_vars))
                else:
                    sess.run(tf.initializers.global_variables())

            start_time = time.time()
            for itr in range(self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                time_env_sampling_start = time.time()

                if self.initial_random_samples and itr < self.num_random_iters:
                    logger.log("Obtaining random samples from the environment...")
                    env_paths = self.sampler.obtain_samples(log=True, random=True, log_prefix='')
                elif self.initial_sinusoid_samples and itr < self.num_random_iters:
                    logger.log("Obtaining sinusoidal samples from the environment...")
                    env_paths = self.sampler.obtain_samples(log=True, log_prefix='', sinusoid=True)
                elif itr % self.on_policy_freq == 0:
                    logger.log("Obtaining samples from the environment using the policy...")
                    env_paths = self.sampler.obtain_samples(log=True, log_prefix='')
                else:
                    logger.log("Obtaining samples from the environment usnig cem policy...")
                    env_paths= self.cem_sampler.obtain_samples(log=True, log_prefix='')

                logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)

                logger.log("Processing environment samples...")

                # first processing just for logging purposes
                time_env_samp_proc = time.time()
                samples_data = self.dynamics_sample_processor.process_samples(env_paths,
                                                                              log=True, log_prefix='EnvTrajs-')
                logger.record_tabular('Time-EnvSampleProc', time.time() - time_env_samp_proc)

                if self.fit_model:

                    ''' --------------- fit dynamics model --------------- '''

                    time_fit_start = time.time()

                    logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
                    self.dynamics_model.fit(samples_data['observations'],
                                            samples_data['actions'],
                                            samples_data['next_observations'],
                                            epochs=self.dynamics_model_max_epochs, verbose=False, log_tabular=True)

                    logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                self.log_diagnostics(env_paths, '')
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
        return dict(itr=itr, policy=self.policy, env=self.env, dynamics_model=self.dynamics_model)

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
