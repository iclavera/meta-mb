import time, sys, pickle
from meta_mb.logger import logger
import multiprocessing


class Worker(object):
    """
    Abstract class for worker instantiations. 
    """
    def __init__(self):
        pass

    def __call__(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict,
            remote,
            synch_notifier=None,
            config=None
    ):
        """
        Args:
            remote (multiprocessing.Connection):
            synch_notifier (multiprocessing.Connection):
        """
        import tensorflow as tf
        with tf.Session(config=config).as_default() as sess:

            self.construct_from_feed_dict(
                policy_pickle,
                env_pickle,
                baseline_pickle,
                dynamics_model_pickle,
                feed_dict
            )

            while True:
                cmd, args = remote.recv()
                print("\n-----------------" + multiprocessing.current_process().name + " starting command " + cmd)

                if cmd == 'step':
                    result_pickle = self.step(args)
                    synch_notifier.send(('synch', result_pickle))

                elif cmd == 'synch':
                    self.synch(args)

                elif cmd == 'close':
                    break

                else:
                    raise NotImplementedError

                print("\n-----------------" + multiprocessing.current_process().name + " finishing command " + cmd)
                sys.stdout.flush()
        sess.close()

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):
        raise NotImplementedError

    def step(self, args=None):
        raise NotImplementedError

    def synch(self, args):
        raise NotImplementedError


class WorkerData(Worker):
    def __init__(self):
        super().__init__()
        self.env = None
        self.env_sampler = None
        self.dynamics_sample_processor = None

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
        self.dynamics_sample_processor = ModelSampleProcessor(baseline=baseline, **feed_dict['dynamics_sample_processor'])

    def step(self, args=None):
        """
        When args is not None, args = initial_random_samples (bool)
        """

        time_env_sampling_start = time.time()

        logger.log("Obtaining samples from the environment using the policy...")
        random = args if args is not None else False
        env_paths = self.env_sampler.obtain_samples(log=True, random=random, log_prefix='EnvSampler-')

#        logger.record_tabular('Time-EnvSampling', time.time() - time_env_sampling_start)
        logger.log("Processing environment samples...")

        # first processing just for logging purposes
        time_env_samp_proc = time.time()
        samples_data = self.dynamics_sample_processor.process_samples(env_paths, log=True,
                                                                      log_prefix='EnvTrajs-')
        self.env.log_diagnostics(env_paths, prefix='EnvTrajs-')
#        logger.record_tabular('Time-EnvSampleProc', time.time() - time_env_samp_proc)
        
        return pickle.dumps(samples_data)

    def synch(self, policy_pickle):
        self.env_sampler.policy = pickle.loads(policy_pickle)
            

class WorkerModel(Worker):
    def __init__(self, dynamics_model_max_epochs):
        super().__init__()
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.dynamics_model = None
        self.samples_data = None

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):
        self.dynamics_model = pickle.loads(dynamics_model_pickle)

    def step(self, args=None):

        assert self.samples_data is not None

        time_fit_start = time.time()

        ''' --------------- fit dynamics model --------------- '''

        logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
        self.dynamics_model.fit(self.samples_data['observations'],
                                self.samples_data['actions'],
                                self.samples_data['next_observations'],
                                epochs=self.dynamics_model_max_epochs, verbose=False, log_tabular=True)

#        logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)

        return pickle.dumps(self.dynamics_model)

    def synch(self, samples_data_pickle):
        self.samples_data = pickle.loads(samples_data_pickle)


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
        from meta_mb.samplers.base import SampleProcessor
        from meta_mb.algos.ppo import PPO

        env = pickle.loads(env_pickle)
        policy = pickle.loads(policy_pickle)
        baseline = pickle.loads(baseline_pickle)

        self.policy = policy
        self.baseline = baseline
        self.model_sampler = METRPOSampler(env=env, policy=policy, **feed_dict['model_sampler'])
        self.model_sample_processor = SampleProcessor(baseline=baseline, **feed_dict['model_sample_processor'])
        self.algo = PPO(policy=policy, **feed_dict['algo'])

    def step(self, args=None):
        """
        Uses self.model_sampler which is asynchrounously updated by worker_model.
        Outcome: policy is updated by PPO on one fictitious trajectory. 
        """

        itr_start_time = time.time()

        """ -------------------- Sampling --------------------------"""

        logger.log("Obtaining samples from the model...")
        time_env_sampling_start = time.time()
        paths = self.model_sampler.obtain_samples(log=True, log_prefix='train-')
        sampling_time = time.time() - time_env_sampling_start

        """ ----------------- Processing Samples ---------------------"""

        logger.log("Processing samples from the model...")
        time_proc_samples_start = time.time()
        samples_data = self.model_sample_processor.process_samples(paths, log='all', log_prefix='train-')
        proc_samples_time = time.time() - time_proc_samples_start

        if type(paths) is list:
            self.log_diagnostics(paths, prefix='train-')
        else:
            self.log_diagnostics(sum(paths.values(), []), prefix='train-')

        """ ------------------ Policy Update ---------------------"""

        logger.log("Optimizing policy...")
        # This needs to take all samples_data so that it can construct graph for meta-optimization.
        time_optimization_step_start = time.time()
        self.algo.optimize_policy(samples_data)
        optimization_time = time.time() - time_optimization_step_start

#        times_dyn_sampling.append(sampling_time)
#        times_dyn_sample_processing.append(proc_samples_time)
#        times_optimization.append(optimization_time)
#        times_step.append(time.time() - itr_start_time)

        return pickle.dumps(self.model_sampler.policy)

    def synch(self, dynamics_model_pickle):
        self.model_sampler.dynamics_model = pickle.loads(dynamics_model_pickle)

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
