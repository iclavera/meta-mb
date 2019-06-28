import time, pickle
import numpy as np
from meta_mb.logger import logger
from meta_mb.workers.worker_policy_base import WorkerPolicyBase


class WorkerPolicy(WorkerPolicyBase):
    def __init__(self, sample_from_buffer, num_inner_grad_steps):
        super().__init__()
        self.sample_from_buffer = sample_from_buffer
        self.num_inner_grad_steps = num_inner_grad_steps

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

    def step(self):
        """
        Uses self.model_sampler which is asynchronously updated by worker_model.
        Outcome: policy is updated by PPO on one fictitious trajectory.
        """

        ''' --------------- MAML steps --------------- '''

        time_maml = np.zeros(3)
        time_step = time.time()

        self.policy.switch_to_pre_update()  # Switch to pre-update policy
        all_samples_data = []

        for step in range(self.num_inner_grad_steps+1):
            if self.verbose:
                logger.log("Policy Adaptation-Step %d **" % step)

            """ -------------------- Sampling --------------------------"""

            time_sampling = time.time()
            paths = self.model_sampler.obtain_samples(log=True, log_prefix='Policy-', buffer=None)
            time_sampling = time.time() - time_sampling

            """ ----------------- Processing Samples ---------------------"""

            time_sample_proc = time.time()
            samples_data = self.model_sample_processor.process_samples(
                paths,
                log='all',
                log_prefix='Policy-'
            )
            all_samples_data.append(samples_data)
            time_sample_proc = time.time() - time_sample_proc

            self.log_diagnostics(sum(list(paths.values()), []), prefix='Policy-')

            """ ------------------- Inner Policy Update --------------------"""

            time_algo_adapt = time.time()
            if step < self.num_inner_grad_steps:
                self.algo._adapt(samples_data)

            time_algo_adapt = time.time() - time_algo_adapt

            time_maml += [time_sampling, time_sample_proc, time_algo_adapt]

        """ ------------------ Outer Policy Update ---------------------"""

        if self.verbose:
            logger.log("Policy is optimizing...")
        # This needs to take all samples_data so that it can construct graph for meta-optimization.
        time_algo_opt = time.time()
        self.algo.optimize_policy(all_samples_data, prefix='Policy-')
        time_algo_opt = time.time() - time_algo_opt

        time_step = time.time() - time_step
        self.result = self.model_sampler.policy
        self.policy = self.result

        info = {
            'Policy-TimeStep': time_step,
            'Policy-TimeInnerSampling': time_maml[0],
            'Policy-TimeInnerSampleProc': time_maml[1],
            'Policy-TimeInnerAlgoAdapt': time_maml[2],
            'Policy-TimeOuterAlgoOpt': time_algo_opt,
        }
        logger.logkvs(info)
