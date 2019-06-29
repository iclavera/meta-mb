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

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict,
    ):

        from meta_mb.samplers.mbmpo_samplers.mbmpo_sampler import MBMPOSampler
        from meta_mb.samplers.meta_samplers.maml_sample_processor import MAMLSampleProcessor
        from meta_mb.meta_algos.trpo_maml import TRPOMAML

        env = pickle.loads(env_pickle)
        policy = pickle.loads(policy_pickle)
        baseline = pickle.loads(baseline_pickle)

        self.policy = policy
        self.baseline = baseline
        self.model_sampler = MBMPOSampler(env=env, policy=policy, **feed_dict['model_sampler'])
        self.model_sample_processor = MAMLSampleProcessor(baseline=baseline, **feed_dict['model_sample_processor'])
        self.algo = TRPOMAML(policy=policy, **feed_dict['algo'])

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

    def dump_result(self):
        self.state_pickle = pickle.dumps(self.result.get_shared_param_values())

    def push(self):
        time_push = time.time()
        self.dump_result()
        put_msg_push = False
        while self.queue_next.qsize() > 3:
            try:
                logger.log('Policy is off loading data from queue_next...')
                data = self.queue_next.get_nowait()
                if data == 'push':
                    put_msg_push = True
            except Empty:
                # very rare chance to reach here (if any)
                break
        self.queue_next.put(self.state_pickle)
        if put_msg_push:
            self.queue_next.put('push')
        time_push = time.time() - time_push

        logger.logkv('Policy-TimePush', time_push)

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
