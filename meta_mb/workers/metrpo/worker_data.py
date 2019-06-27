import time, pickle
from meta_mb.logger import logger
from meta_mb.workers.base import Worker

class WorkerData(Worker):
    def __init__(self, simulation_sleep):
        super().__init__()
        self.simulation_sleep = simulation_sleep
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
        # compute window size in terms of number of steps
        # self.window_size = np.ceil(self.window_size / np.prod(feed_dict['env_sampler'].values()))

    def prepare_start(self):
        initial_random_samples = self.queue.get()
        self.step(initial_random_samples)
        self.queue_next.put(pickle.dumps(self.result))

    def step(self, random=False):
        """
        When args is not None, args = initial_random_samples (bool)
        """

        '''------------- Obtaining samples from the environment -----------'''

        if self.verbose:
            logger.log("Obtaining samples from the environment...")
        time_env_sampling = time.time()
        env_paths = self.env_sampler.obtain_samples(
            log=True,
            random=random,
            log_prefix='Data-EnvSampler-',
        )
        time_env_sampling = time.time() - time_env_sampling

        '''-------------- Processing environment samples -------------------'''

        if self.verbose:
            logger.log("Processing environment samples...")
        # first processing just for logging purposes
        time_env_samp_proc = time.time()
        samples_data = self.dynamics_sample_processor.process_samples(
            env_paths,
            log=True,
            log_prefix='Data-EnvTrajs-',
        )
        time_env_samp_proc = time.time() - time_env_samp_proc

        time.sleep(self.simulation_sleep)
        self.result = samples_data

        info = {'Data-Iteration': self.itr_counter,
                'Data-TimeEnvSampling': time_env_sampling, 'Data-TimeEnvSampProc': time_env_samp_proc}
        logger.logkvs(info)

    def _synch(self, policy_state_pickle):
        time_synch = time.time()
        policy_state = pickle.loads(policy_state_pickle)
        assert isinstance(policy_state, dict)
        self.env_sampler.policy.set_shared_params(policy_state)
        time_synch = time.time() - time_synch

        logger.logkv('Data-TimeSynch', time_synch)

    def push(self):
        time_push = time.time()
        self.dump_result()
        self.queue_next.put(self.state_pickle)
        time_push = time.time() - time_push
        logger.logkv('Data-TimePush', time_push)

    def set_stop_cond(self):
        if self.itr_counter >= self.n_itr:
            self.stop_cond.set()

    # similar to log_real_performance
    def prepare_close(self, data):
        # step one more time with most updated policy to measure performance
        # result dumped in logger
        raise NotImplementedError
