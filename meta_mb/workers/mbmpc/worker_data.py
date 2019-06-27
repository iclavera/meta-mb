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
            baseline_pickle,  # UNUSED
            dynamics_model_pickle,
            feed_dict,
    ):

        from meta_mb.samplers.sampler import Sampler
        from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor

        env = pickle.loads(env_pickle)
        policy = pickle.loads(policy_pickle)

        self.env = env
        self.env_sampler = Sampler(env=env, policy=policy, **feed_dict['sampler'])
        self.dynamics_sample_processor = ModelSampleProcessor(
            **feed_dict['sample_processor']
        )

    def prepare_start(self):
        initial_random_samples, initial_sinusoid_samples = self.queue.get()
        self.step(initial_random_samples, initial_sinusoid_samples)
        self.queue_next.put(pickle.dumps(self.result))

    def step(self, random=False, sinusoid=False):

        if self.itr_counter == 1 and self.env_sampler.policy.dynamics_model.normalization is None:
            if self.verbose:
                logger.log('Data starts first step...')
            self.env_sampler.policy.dynamics_model = pickle.loads(self.queue.get())
            if self.verbose:
                logger.log('Data first step done...')

        '''------------- Obtaining samples from the environment -----------'''

        if self.verbose:
            logger.log("Data is obtaining samples...")
        time_env_sampling = time.time()
        env_paths = self.env_sampler.obtain_samples(
            log=True,
            random=random,
            sinusoid=sinusoid,
            log_prefix='Data-EnvSampler-',
        )
        time_env_sampling = time.time() - time_env_sampling

        '''-------------- Processing environment samples -------------------'''

        if self.verbose:
            logger.log("Data is processing samples...")
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

    def _synch(self, dynamics_model_state_pickle):
        time_synch = time.time()
        dynamics_model_state = pickle.loads(dynamics_model_state_pickle)
        assert isinstance(dynamics_model_state, dict)
        self.env_sampler.policy.dynamics_model.set_shared_params(dynamics_model_state)
        time_synch = time.time() - time_synch

        logger.logkv('Data-TimeSynch', time_synch)

    def set_stop_cond(self):
        if self.itr_counter >= self.n_itr:
            self.stop_cond.set()

    # similar to log_real_performance
    def prepare_close(self, data):
        # step one more time with most updated policy to measure performance
        # result dumped in logger
        raise NotImplementedError
