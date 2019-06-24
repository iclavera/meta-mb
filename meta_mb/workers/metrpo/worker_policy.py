import time, pickle
import numpy as np
from meta_mb.logger import logger
from meta_mb.workers.base import Worker


class WorkerPolicy(Worker):
    def __init__(self, step_per_iter):
        super().__init__()
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
        from meta_mb.samplers.bptt_samplers.bptt_sampler import BPTTSampler
        from meta_mb.samplers.base import SampleProcessor
        from meta_mb.algos.ppo import PPO
        print('start construction...')

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
            paths = self.model_sampler.obtain_samples(log=True, log_prefix='Policy-')

            for path in paths:
                assert not np.isnan(np.sum(path['observations']))
                assert not np.isnan(np.sum(path['actions']))
                assert not np.isnan(np.sum(path['rewards']))
            time_sampling = time.time() - time_sampling

            """ ----------------- Processing Samples ---------------------"""

            if self.verbose:
                logger.log("Processing samples from the model...")
            time_sample_proc = time.time()
            samples_data = self.model_sample_processor.process_samples(
                paths,
                log='all',
                log_prefix='Policy-'
            )
            time_sample_proc = time.time() - time_sample_proc

            if type(paths) is list:
                self.log_diagnostics(paths, prefix='Policy-')
            else:
                self.log_diagnostics(sum(paths.values(), []), prefix='Policy-')

            """ ------------------ Policy Update ---------------------"""

            if self.verbose:
                logger.log("Optimizing policy...")
            # This needs to take all samples_data so that it can construct graph for meta-optimization.
            time_algo_opt = time.time()
            self.algo.optimize_policy(samples_data, log=True, verbose=self.verbose)
            time_algo_opt = time.time() - time_algo_opt

            time_step = time.time() - time_step
            time_maml += [time_step, time_sampling, time_sample_proc, time_algo_opt]

        self.result = self.model_sampler.policy
        self.policy = self.result

        self.update_info()
        info = {'Policy-Iteration': self.itr_counter,
                'Policy-TimeStep': time_maml[0], 'Policy-TimeSampling': time_maml[1],
                'Policy-TimeSampleProc': time_maml[2], 'Policy-TimeAlgoOpt': time_maml[3], }
        self.info.update(info)

    def _synch(self, dynamics_model_state_pickle):
        # time_synch = time.time()
        if self.verbose:
            logger.log('policy is synchronizing...')
        dynamics_model_state = pickle.loads(dynamics_model_state_pickle)
        assert isinstance(dynamics_model_state, dict)
        self.model_sampler.dynamics_model.set_shared_params(dynamics_model_state)
        self.model_sampler.vec_env.dynamics_model.set_shared_params(dynamics_model_state)
        if self.verbose:
            logger.log('policy quits synch...')
        # time_synch = time.time() - time_synch
        # info = {'Policy-TimeSynch': time_synch}
        # self.info.update(info)

    def dump_result(self):
        self.state_pickle = pickle.dumps(self.result.get_shared_param_values())

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
