import time, pickle
from meta_mb.logger import logger
from meta_mb.workers.base import Worker


class WorkerData(Worker):
    def __init__(self, time_sleep, video=False):
        self.video = video
        if video:
            super().__init__(time_sleep=time_sleep, snapshot_mode='gap', snapshot_gap=100)
        else:
            super().__init__(time_sleep=time_sleep)
        self.env = None
        self.env_sampler = None
        self.dynamics_sample_processor = None
        self.samples_data_arr = []

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

    def prepare_start(self):
        initial_random_samples = self.queue.get()
        self.step(initial_random_samples)
        self.push()

    def step(self, random=False):
        time_step = time.time()

        '''------------- Obtaining samples from the environment -----------'''

        if self.verbose:
            logger.log("Data is obtaining samples...")
        env_paths = self.env_sampler.obtain_samples(
            log=True,
            random=random,
            log_prefix='Data-EnvSampler-',
        )

        '''-------------- Processing environment samples -------------------'''

        if self.verbose:
            logger.log("Data is processing environment samples...")
        samples_data = self.dynamics_sample_processor.process_samples(
            env_paths,
            log=True,
            log_prefix='Data-EnvTrajs-',
        )

        self.samples_data_arr.append(samples_data)
        time_step = time.time() - time_step

        time_sleep = max(self.time_sleep - time_step, 0)
        time.sleep(time_sleep)

        logger.logkv('Data-TimeStep', time_step)
        logger.logkv('Data-TimeSleep', time_sleep)

        # save snapshot
        if self.video and self.itr_counter % self.snapshot_gap == 0:
            time_save = time.time()
            params = self.get_itr_snapshot()
            logger.save_itr_params(self.itr_counter, params)
            logger.logkv('Data-TimeSave', time.time() - time_save)

    def _synch(self, policy_state_pickle):
        time_synch = time.time()
        policy_state = pickle.loads(policy_state_pickle)
        assert isinstance(policy_state, dict)
        self.env_sampler.policy.set_shared_params(policy_state)
        time_synch = time.time() - time_synch

        logger.logkv('Data-TimeSynch', time_synch)

    def push(self):
        time_push = time.time()
        self.queue_next.put(pickle.dumps(self.samples_data_arr))
        self.samples_data_arr = []
        time_push = time.time() - time_push

        logger.logkv('Data-TimePush', time_push)

    def set_stop_cond(self):
        if self.itr_counter >= self.n_itr:
            self.stop_cond.set()

    def get_itr_snapshot(self):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=self.itr_counter, policy=self.env_sampler.policy, env=self.env)
