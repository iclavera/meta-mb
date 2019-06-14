import time, pickle
import numpy as np
from meta_mb.logger import logger
from multiprocessing import current_process
from queue import Empty


class Worker(object):
    """
    Abstract class for worker instantiations. 
    """
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict,
            queue,
            queue_next,
            remote,
            start_itr,
            n_itr,
            stop_cond,
            config=None,
    ):
        """
        Args:
            queue (multiprocessing.Queue): queue for current worker
            queue_next (multiprocessing.Queue): queue for next worker
            rcp_sender (multiprocessing.Connection): notify scheduler after task completed
        """
        self.n_itr = n_itr

        import tensorflow as tf

        def _init_vars():
            sess = tf.get_default_session()
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

        with tf.Session(config=config).as_default() as _:

            self.construct_from_feed_dict(
                policy_pickle,
                env_pickle,
                baseline_pickle,
                dynamics_model_pickle,
                feed_dict,
            )

            _init_vars()
            steps_per_synch = []
            step_per_synch = 0

            # warm up
            self.itr_counter = start_itr
            assert remote.recv() == 'prepare start'
            data = queue.get()
            result_pickle = self.prepare_start(data)
            logger.dumpkvs()
            step_per_synch += 1
            remote.send('loop ready')
            queue_next.put(result_pickle)
            logger.log("\n============== {} is ready =============".format(current_process().name))

            assert remote.recv() == 'start loop'
            time_start = time.time()
            while not stop_cond.is_set():
                data = None
                # only the latest data is used for synchronization
                while True:
                    try:
                        data = queue.get_nowait()
                    except Empty:
                        break
                if data is not None:
                    steps_per_synch.append(step_per_synch)
                    step_per_synch = 0
                    action = 'synch'
                    self.synch(data)
                else:
                    self.itr_counter += 1
                    step_per_synch += 1
                    action = 'step {}'.format(self.itr_counter)
                    logger.logkv('Worker', current_process().name)
                    logger.logkv('Iteration', self.itr_counter)
                    logger.logkv('StepPerSynch', step_per_synch)
                    result_pickle = self.step()
                    queue_next.put(result_pickle)
                    logger.dumpkvs()

                remote.send(action)
                logger.log("\n========================== {} completes {} ===================".format(
                    current_process().name, action
                ))
                if self.set_stop_cond():
                    stop_cond.set()

            remote.send('loop done')

            # Alternatively, to avoid repetitive code chunk, let scheduler send latest data
            data = None
            while True:
                try:
                    data = queue.get_nowait()
                except Empty:
                    break
            assert queue.empty()
            self.prepare_close(data)
            logger.log("\n========== prepared close =====================")
            remote.send('worker closed')

        logger.logkv('TimeTotal', time.time() - time_start)
        logger.logkv('Worker', current_process().name)
        logger.logkv('StepPerSynch', np.mean(steps_per_synch))
        logger.dumpkvs()
        logger.log("\n================== {} closed ===================".format(
            current_process().name
        ))

        remote.send('worker closed')

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):
        raise NotImplementedError

    def prepare_start(self, args):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def synch(self, args):
        raise NotImplementedError

    def set_stop_cond(self):
        return False

    def prepare_close(self, args):
        pass


class WorkerData(Worker):
    def __init__(self):
        super().__init__()
        self.env = None
        self.env_sampler = None
        self.dynamics_sample_processor = None

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):

        from meta_mb.samplers.sampler import Sampler
        from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor

        env = pickle.loads(env_pickle)
        policy = pickle.loads(policy_pickle)
        baseline = pickle.loads(baseline_pickle)

        self.env = env
        self.env_sampler = Sampler(env=env, policy=policy, **feed_dict['env_sampler'])
        self.dynamics_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            **feed_dict['dynamics_sample_processor']
        )

    def prepare_start(self, initial_random_samples):
        return self.step(initial_random_samples)

    def step(self, random=False):
        """
        When args is not None, args = initial_random_samples (bool)
        """

        if self.verbose:
            logger.log("Obtaining samples from the environment using the policy...")
        time_env_sampling = time.time()
        env_paths = self.env_sampler.obtain_samples(
            log=True,
            random=random,
            log_prefix='EnvSampler-',
            verbose=self.verbose,
        )
        time_env_sampling = time.time() - time_env_sampling

        if self.verbose:
            logger.log("Processing environment samples...")
        # first processing just for logging purposes
        time_env_samp_proc = time.time()
        samples_data = self.dynamics_sample_processor.process_samples(
            env_paths,
            log=True,
            log_prefix='EnvTrajs-'
        )
        self.env.log_diagnostics(env_paths, prefix='EnvTrajs-')
        time_env_samp_proc = time.time() - time_env_samp_proc

        # info = [time_env_sampling, time_env_samp_proc]
        logger.logkv("TimeEnvSampling", time_env_sampling)
        logger.logkv("TimeEnvSampProc", time_env_samp_proc)

        time.sleep(10)

        return pickle.dumps(samples_data)

    def synch(self, policy_pickle):
        self.env_sampler.policy = pickle.loads(policy_pickle)

    def prepare_close(self, data):
        # step one more time with most updated policy to measure performance
        # result dumped in logger
        self.synch(data)
        _ = self.step()


class WorkerModel(Worker):
    def __init__(self, dynamics_model_max_epochs, warm_next=True):
        super().__init__(warm_next)
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.dynamics_model = None
        self.samples_data = None

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):
        self.dynamics_model = pickle.loads(dynamics_model_pickle)

    def prepare_start(self, data):
        self.synch(data)
        return self.step()

    def step(self):

        assert self.samples_data is not None

        time_model_fit = time.time()

        ''' --------------- fit dynamics model --------------- '''

        if self.verbose:
            logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
        self.dynamics_model.fit(self.samples_data['observations'],
                                self.samples_data['actions'],
                                self.samples_data['next_observations'],
                                epochs=self.dynamics_model_max_epochs, verbose=False, log_tabular=True)

        time_model_fit = time.time() - time_model_fit

        # info = [time_model_fit]
        logger.logkv("TimeModelFit", time_model_fit)

        return pickle.dumps(self.dynamics_model)

    def synch(self, samples_data_pickle):
        self.samples_data = pickle.loads(samples_data_pickle)


class WorkerPolicy(Worker):
    def __init__(self, step_per_iter, warm_next=False):
        super().__init__(warm_next)
        self.step_per_iter = step_per_iter
        self.policy = None
        self.baseline = None
        self.model_sampler = None
        self.model_sample_processor = None
        self.algo = None

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):

        from meta_mb.samplers.metrpo_samplers.metrpo_sampler import METRPOSampler
        from meta_mb.samplers.base import SampleProcessor
        from meta_mb.algos.ppo import PPO

        env = pickle.loads(env_pickle)
        policy = pickle.loads(policy_pickle)
        baseline = pickle.loads(baseline_pickle)

        self.policy = policy
        self.baseline = baseline
        self.model_sampler = METRPOSampler(env=env, policy=policy, **feed_dict['model_sampler'])
        self.model_sample_processor = SampleProcessor(baseline=baseline, **feed_dict['model_sample_processor'])
        self.algo = PPO(policy=policy, **feed_dict['algo'])

    def prepare_start(self, args):
        self.synch(args)
        return self.step()

    def step(self):
        """
        Uses self.model_sampler which is asynchronously updated by worker_model.
        Outcome: policy is updated by PPO on one fictitious trajectory. 
        """

        ''' --------------- MAML steps --------------- '''

        time_maml = np.zeros(4)

        for step in range(self.step_per_iter):

            time_step = time.time()

            """ -------------------- Sampling --------------------------"""

            if self.verbose:
                logger.log("Obtaining samples from the model...")
            time_sampling = time.time()
            paths = self.model_sampler.obtain_samples(log=False, log_prefix='train-')
            time_sampling = time.time() - time_sampling

            """ ----------------- Processing Samples ---------------------"""

            if self.verbose:
                logger.log("Processing samples from the model...")
            time_sample_proc = time.time()
            samples_data = self.model_sample_processor.process_samples(
                paths,
                log='all',
                log_prefix='train-'
            )
            time_sample_proc = time.time() - time_sample_proc

            if type(paths) is list:
                self.log_diagnostics(paths, prefix='train-')
            else:
                self.log_diagnostics(sum(paths.values(), []), prefix='train-')

            """ ------------------ Policy Update ---------------------"""

            if self.verbose:
                logger.log("Optimizing policy...")
            # This needs to take all samples_data so that it can construct graph for meta-optimization.
            time_algo_opt = time.time()
            self.algo.optimize_policy(samples_data, log=True, verbose=self.verbose)
            time_algo_opt = time.time() - time_algo_opt

            time_step = time.time() - time_step
            time_maml += [time_step, time_sampling, time_sample_proc, time_algo_opt]

        # info = [avg_time_step, avg_time_sampling, avg_time_sample_proc, avg_algo_opt]
        for key, val in zip(['TimeAvgStep', 'TimeAvgSampling', 'TimeAvgSampleProc', 'TimeAvgAlgoOpt'], time_maml):
            logger.logkv(key, val/self.step_per_iter)

        return pickle.dumps(self.model_sampler.policy)

    def synch(self, dynamics_model_pickle):
        dynamics_model = pickle.loads(dynamics_model_pickle)
        self.model_sampler.dynamics_model = dynamics_model
        self.model_sampler.vec_env.dynamics_model = dynamics_model

    def set_stop_cond(self):
        return self.itr_counter >= self.n_itr

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
