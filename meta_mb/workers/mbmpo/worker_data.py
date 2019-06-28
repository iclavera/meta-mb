import time, pickle
import numpy as np
from meta_mb.logger import logger
from meta_mb.workers.worker_data_base import WorkerDataBase


class WorkerData(WorkerDataBase):
    def __init__(self, fraction_meta_batch_size, simulation_sleep):
        super().__init__(simulation_sleep)
        self.fraction_meta_batch_size = fraction_meta_batch_size

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):

        # from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
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

    def step(self, random=False):

        time_step = time.time()

        '''------------- Obtaining samples from the environment -----------'''

        if self.verbose:
            logger.log("Data is obtaining samples...")
        time_env_sampling = time.time()
        env_paths = self.env_sampler.obtain_samples(
            log=True,
            random=random,
            log_prefix='Data-EnvSampler-',
        )
        time_env_sampling = time.time() - time_env_sampling

        '''-------------- Processing environment samples -------------------'''

        if self.verbose:
            logger.log("Data is processing samples...")
        time_env_samp_proc = time.time()
        """
        env_paths = list(env_paths.values())
        idxs = np.random.choice(
            range(len(env_paths)),
            size=int(len(env_paths)* self.fraction_meta_batch_size),
            replace=False,
        )
        env_paths = sum([env_paths[idx] for idx in idxs], [])
        """
        samples_data = self.dynamics_sample_processor.process_samples(
            env_paths,
            log=True,
            log_prefix='Data-EnvTrajs-',
        )
        time_env_samp_proc = time.time() - time_env_samp_proc

        if self.result is None:
            self.result = samples_data
        else:
            self.result['actions'] = np.append(self.result['actions'], samples_data['actions'], axis=0)
            self.result['observations'] = np.append(self.result['observations'], samples_data['observations'], axis=0)
            self.result['next_observations'] = np.append(self.result['next_observations'], samples_data['next_observations'], axis=0)

        time_step = time.time() - time_step
        time.sleep(self.simulation_sleep)

        logger.logkv('Data-TimeStep', time_step)
        logger.logkv('Data-TimeEnvSampling', time_env_sampling)
        logger.logkv('Data-TimeEnvSampProc', time_env_samp_proc)


