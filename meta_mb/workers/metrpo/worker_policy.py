import time, pickle
from queue import Empty
from meta_mb.logger import logger
from meta_mb.workers.base import Worker


class WorkerPolicy(Worker):
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
            feed_dict
    ):

        from meta_mb.samplers.metrpo_samplers.metrpo_sampler import METRPOSampler
        from meta_mb.samplers.bptt_samplers.bptt_sampler import BPTTSampler
        from meta_mb.samplers.base import SampleProcessor
        from meta_mb.algos.ppo import PPO

        env = pickle.loads(env_pickle)
        policy = pickle.loads(policy_pickle)
        baseline = pickle.loads(baseline_pickle)

        self.policy = policy
        self.baseline = baseline
        self.model_sampler = METRPOSampler(env=env, policy=policy, **feed_dict['model_sampler'])
        # self.model_sampler = BPTTSampler(env=env, policy=policy, **feed_dict['model_sampler'])
        self.model_sample_processor = SampleProcessor(baseline=baseline, **feed_dict['model_sample_processor'])
        self.algo = PPO(policy=policy, **feed_dict['algo'])

    def prepare_start(self):
        dynamics_model = pickle.loads(self.queue.get())
        self.model_sampler.dynamics_model = dynamics_model
        self.model_sampler.vec_env.dynamics_model = dynamics_model
        self.step()
        # self.queue_next.put(pickle.dumps(self.result))
        self.push()

    def step(self):
        time_step = time.time()

        """ -------------------- Sampling --------------------------"""

        if self.verbose:
            logger.log("Policy is obtaining samples ...")
        #time_sampling = time.time()
        paths = self.model_sampler.obtain_samples(log=True, log_prefix='Policy-')
        #time_sampling = time.time() - time_sampling

        """ ----------------- Processing Samples ---------------------"""

        #time_sample_proc = time.time()
        if self.verbose:
            logger.log("Policy is processing samples ...")
        samples_data = self.model_sample_processor.process_samples(
            paths,
            log='all',
            log_prefix='Policy-'
        )
        #time_sample_proc = time.time() - time_sample_proc

        if type(paths) is list:
            self.log_diagnostics(paths, prefix='Policy-')
        else:
            self.log_diagnostics(sum(paths.values(), []), prefix='Policy-')

        """ ------------------ Policy Update ---------------------"""

        #time_algo_opt = time.time()
        if self.verbose:
            logger.log("Policy optimization...")
        # This needs to take all samples_data so that it can construct graph for meta-optimization.
        self.algo.optimize_policy(samples_data, log=True, verbose=self.verbose, prefix='Policy-')
        #time_algo_opt = time.time() - time_algo_opt

        self.policy = self.model_sampler.policy
        time_step = time.time() - time_step

        logger.logkv('Policy-TimeStep', time_step)
        #logger.logkv('Policy-TimeSampling', time_sampling)
        #logger.logkv('Policy-TimeSampleProc', time_sample_proc)
        #logger.logkv('Policy-TimeAlgoOpt', time_algo_opt)

    def _synch(self, dynamics_model_state_pickle):
        time_synch = time.time()
        if self.verbose:
            logger.log('Policy is synchronizing...')
        dynamics_model_state = pickle.loads(dynamics_model_state_pickle)
        assert isinstance(dynamics_model_state, dict)
        self.model_sampler.dynamics_model.set_shared_params(dynamics_model_state)
        self.model_sampler.vec_env.dynamics_model.set_shared_params(dynamics_model_state)
        time_synch = time.time() - time_synch

        logger.logkv('Policy-TimeSynch', time_synch)

    def push(self):
        time_push = time.time()
        policy_state_pickle = pickle.dumps(self.policy.get_shared_param_values())
        assert policy_state_pickle is not None
        while self.queue_next.qsize() > 5:
            try:
                logger.log('Policy is off loading data from queue_next...')
                _ = self.queue_next.get_nowait()
            except Empty:
                # very rare chance to reach here (if any)
                break
        self.queue_next.put(policy_state_pickle)
        time_push = time.time() - time_push

        logger.logkv('Policy-TimePush', time_push)

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
