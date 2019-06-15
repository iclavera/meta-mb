import time, pickle
import numpy as np
from meta_mb.logger import logger
from multiprocessing import current_process
from queue import Empty


class Worker(object):
    """
    Abstract class for worker instantiations. 
    """
    def __init__(
            self, 
            verbose=False, 
    ):
        self.verbose = verbose
        self.result_pickle = None

    def __call__(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict,
            queue_prev, 
            queue, 
            queue_next, 
            remote,
            start_itr,
            n_itr,
            stop_cond,
            need_query=False, 
            auto_push=True, 
            config=None,
    ):
        """
        Args:
            queue (multiprocessing.Queue): queue for current worker
            queue_next (multiprocessing.Queue): queue for next worker
            rcp_sender (multiprocessing.Connection): notify scheduler after task completed
        """
        self.n_itr = n_itr
        self.queue = queue

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

            # warm up
            self.itr_counter = start_itr
            assert remote.recv() == 'prepare start'
            data = self.queue.get()
            self.prepare_start(data)
            logger.dumpkvs()
            remote.send('loop ready')
            queue_next.put(self.result_pickle)
            logger.log("\n============== {} is ready =============".format(current_process().name))

            assert remote.recv() == 'start loop'
            time_start = time.time()
            while not stop_cond.is_set():
                if need_query: # poll
                    queue_prev.put('push')
                do_push, do_step = self.process_queue()
                if do_push: # push
                    queue_next.put(self.result_pickle)
                if do_step: # step
                    self.itr_counter += 1
                    logger.logkv('Worker', current_process().name)
                    logger.logkv('Iteration', self.itr_counter)
                    self.step()
                    logger.dumpkvs()
                    if auto_push:
                        queue_next.put(self.result_pickle)

                remote.send((int(do_push), int(do_step)))
                logger.log("\n========================== {} {} {} ===================".format(
                    current_process().name,
                    ('push' if do_push else None, 'step' if do_step else 'synch'),
                    self.itr_counter
                ))
                if self.set_stop_cond():
                    stop_cond.set()

            remote.send('loop done')

            # worker_policy push latest policy
            # worker_data synch and step

            # Alternatively, to avoid repetitive code chunk, let scheduler send latest data
            # FIXME
            """
            data = None
            while True:
                try:
                    data = queue.get_nowait()
                except Empty:
                    break
            assert queue.empty()
            self.prepare_close(data)
            logger.log("\n========== prepared close =====================")
            """
            remote.send('worker closed')

        logger.logkv('TimeTotal', time.time() - time_start)
        logger.logkv('Worker', current_process().name)
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
    
    def process_queue(self):
        push = False
        data = None
        while True:
            try:
                new_data = self.queue.get_nowait()
                if new_data == 'push':
                    push = True
                else:
                    data = new_data
            except Empty:
                break

        # actions is a boolean array
        # actions[0] = push (True) or not (False)
        # actions[1] = step (True) or synch (False)
        actions = [push, data is None]
        if data is not None:
            self._synch(data)
        return actions
    
    def _synch(self, data):
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
        self.step(initial_random_samples)

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

        self.result_pickle = pickle.dumps(samples_data)

    def _synch(self, policy_pickle):
        self.env_sampler.policy = pickle.loads(policy_pickle)

    def set_stop_cond(self):
        return self.itr_counter >= self.n_itr

    def prepare_close(self, data):
        # step one more time with most updated policy to measure performance
        # result dumped in logger
        raise NotImplementedError

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
        self._synch(data)
        self.step()

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

        self.result_pickle = pickle.dumps(self.dynamics_model)

    def _synch(self, samples_data_pickle):
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

    def prepare_start(self, data):
        self._synch(data)
        self.step()

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

        self.result_pickle = pickle.dumps(self.model_sampler.policy)

    def _synch(self, dynamics_model_pickle):
        dynamics_model = pickle.loads(dynamics_model_pickle)
        self.model_sampler.dynamics_model = dynamics_model
        self.model_sampler.vec_env.dynamics_model = dynamics_model

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
