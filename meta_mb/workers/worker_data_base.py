import time
import pickle
import numpy as np
from meta_mb.logger import logger
from meta_mb.workers.base import Worker


class WorkerDataBase(Worker):
    def __init__(self, simulation_sleep, *args, **kwargs):
        super().__init__()
        self.simulation_sleep = simulation_sleep
        self.env = None
        self.env_sampler = None
        self.dynamics_sample_processor = None
        self.result = []

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):
        raise NotImplementedError

    def prepare_start(self):
        self.step(self.queue.get())
        self.push()

    def step(self, random=False):
        raise NotImplementedError

    def push(self):
        time_push = time.time()
        self.dump_result()
        self.queue_next.put(self.state_pickle)
        time_push = time.time() - time_push
        logger.logkv('Data-TimePush', time_push)

    def set_stop_cond(self):
        if self.itr_counter >= self.n_itr:
            self.stop_cond.set()

    def _synch(self, policy_state_pickle):
        time_synch = time.time()
        policy_state = pickle.loads(policy_state_pickle)
        self.env_sampler.policy.set_shared_params(policy_state)
        time_synch = time.time() - time_synch

        logger.logkv('Data-TimeSynch', time_synch)

    def dump_result(self):
        act = np.concatenate([samples_data['actions'] for samples_data in self.result])
        obs = np.concatenate([samples_data['observations'] for samples_data in self.result])
        obs_next = np.concatenate([samples_data['next_observations'] for samples_data in self.result])
        self.result = []
        self.state_pickle = pickle.dumps((act, obs, obs_next))
