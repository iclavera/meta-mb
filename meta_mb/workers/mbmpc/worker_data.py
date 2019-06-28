import time, pickle
import numpy as np
from meta_mb.logger import logger
from meta_mb.workers.worker_data_base import WorkerDataBase


class WorkerData(WorkerDataBase):
    def __init__(self, simulation_sleep):
        super().__init__(simulation_sleep)

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

    def step(self, random_sinusoid=(False, False)):

        time_step = time.time()

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
            random=random_sinusoid[0],
            sinusoid=random_sinusoid[1],
            log_prefix='Data-EnvSampler-',
        )
        time_env_sampling = time.time() - time_env_sampling

        '''-------------- Processing environment samples -------------------'''

        if self.verbose:
            logger.log("Data is processing samples...")
        time_env_samp_proc = time.time()
        samples_data = self.dynamics_sample_processor.process_samples(
            env_paths,
            log=True,
            log_prefix='Data-EnvTrajs-',
        )
        time_env_samp_proc = time.time() - time_env_samp_proc

        if self.result is None:
            self.result = samples_data
        else:
            self.result['actions'] = np.concatenate([
                self.result['actions'], samples_data['actions']
            ])
            self.result['observations'] = np.concatenate([
                self.result['observations'], samples_data['observations']
            ])
            self.result['next_observations'] = np.concatenate([
                self.result['next_observations'], samples_data['next_observations']
            ])

        time_step = time.time() - time_step
        time.sleep(max(self.simulation_sleep - time_step, 0))

        logger.logkv('Data-TimeStep', time_step)
        logger.logkv('Data-TimeEnvSampling', time_env_sampling)
        logger.logkv('Data-TimeEnvSampProc', time_env_samp_proc)

    def _synch(self, dynamics_model_state_pickle):
        time_synch = time.time()
        dynamics_model_state = pickle.loads(dynamics_model_state_pickle)
        assert isinstance(dynamics_model_state, dict)
        self.env_sampler.policy.dynamics_model.set_shared_params(dynamics_model_state)
        time_synch = time.time() - time_synch

        logger.logkv('Data-TimeSynch', time_synch)

