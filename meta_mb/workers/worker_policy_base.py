import time, pickle
from meta_mb.logger import logger
from meta_mb.workers.base import Worker
from queue import Empty


class WorkerPolicyBase(Worker):
    def __init__(self):
        super().__init__()
        self.policy = None
        self.baseline = None
        self.model_sampler = None
        self.model_sample_processor = None
        self.algo = None

    def prepare_start(self):
        dynamics_model = pickle.loads(self.queue.get())
        self.model_sampler.dynamics_model = dynamics_model
        self.model_sampler.vec_env.dynamics_model = dynamics_model
        self.step()
        self.push()

    def step(self):
        raise NotImplementedError

    def _synch(self, dynamics_model_state_pickle):
        time_synch = time.time()
        dynamics_model_state = pickle.loads(dynamics_model_state_pickle)
        assert isinstance(dynamics_model_state, dict)
        self.model_sampler.dynamics_model.set_shared_params(dynamics_model_state)
        self.model_sampler.vec_env.dynamics_model.set_shared_params(dynamics_model_state)
        time_synch = time.time() - time_synch

        logger.logkv('Policy-TimeSynch', time_synch)

    def push(self):
        time_push = time.time()
        policy_state_pickle = pickle.dumps(self.policy.get_shared_param_values())
        while self.queue_next.qsize() > 3:
            try:
                logger.log('Policy is off loading data from queue_next...')
                _ = self.queue_next.get_nowait()
            except Empty:
                # very rare chance to reach here (if any)
                break
        assert policy_state_pickle is not None
        self.queue_next.put(policy_state_pickle)
        time_push = time.time() - time_push

        logger.logkv('Policy-TimePush', time_push)

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
