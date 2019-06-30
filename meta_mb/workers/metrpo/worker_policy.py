import time, pickle
import numpy as np
from meta_mb.logger import logger
from meta_mb.workers.worker_policy_base import WorkerPolicyBase


class WorkerPolicy(WorkerPolicyBase):
    def __init__(self, algo_str):
        super().__init__()
        self.algo_str = algo_str

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
        from meta_mb.algos.trpo import TRPO

        env = pickle.loads(env_pickle)
        policy = pickle.loads(policy_pickle)
        baseline = pickle.loads(baseline_pickle)

        self.policy = policy
        self.baseline = baseline
        self.model_sampler = METRPOSampler(env=env, policy=policy, **feed_dict['model_sampler'])
        # self.model_sampler = BPTTSampler(env=env, policy=policy, **feed_dict['model_sampler'])
        self.model_sample_processor = SampleProcessor(baseline=baseline, **feed_dict['model_sample_processor'])
        if self.algo_str == 'meppo':
            self.algo = PPO(policy=policy, **feed_dict['algo'])
        elif self.algo_str == 'metrpo':
            self.algo = TRPO(policy=policy, **feed_dict['algo'])
        else:
            raise NotImplementedError('algo_str must be meppo or metrpo, got {}'.format(self.algo_str))

    def step(self):
        """
        Uses self.model_sampler which is asynchronously updated by worker_model.
        Outcome: policy is updated by PPO on one fictitious trajectory.
        """

        ''' --------------- MAML steps --------------- '''

        time_step = time.time()

        """ -------------------- Sampling --------------------------"""

        if self.verbose:
            logger.log("Policy is obtaining samples ...")
        time_sampling = time.time()
        paths = self.model_sampler.obtain_samples(log=True, log_prefix='Policy-')
        time_sampling = time.time() - time_sampling

        """ ----------------- Processing Samples ---------------------"""

        if self.verbose:
            logger.log("Policy is processing samples ...")
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
            logger.log("Policy optimization...")
        # This needs to take all samples_data so that it can construct graph for meta-optimization.
        time_algo_opt = time.time()
        self.algo.optimize_policy(samples_data, log=True, prefix='Policy-', verbose=self.verbose)
        time_algo_opt = time.time() - time_algo_opt

        time_step = time.time() - time_step
        self.result = self.model_sampler.policy
        self.policy = self.result

        info = {
            'Policy-TimeStep': time_step,
            'Policy-TimeSampling': time_sampling,
            'Policy-TimeSampleProc': time_sample_proc,
            'Policy-TimeAlgoOpt': time_algo_opt,
        }
        logger.logkvs(info)

