import time, sys, pickle
from meta_mb.logger import logger
import multiprocessing


class Worker(object):
    """
    Abstract class for worker instantiations. 
    """
    def __init__(self):
        pass

    def __call__(self, remote, synch_notifier=None, config=None):
        """
        Args:
            remote (multiprocessing.Connection):
            synch_notifier (multiprocessing.Connection):
        """
        import tensorflow as tf
        with tf.Session(config=config).as_default() as sess:
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

    def step(self, args=None):
        raise NotImplementedError

    def synch(self, args):
        raise NotImplementedError


class WorkerData(Worker):
    def __init__(
            self,
            env,
            env_sampler, 
            dynamics_sample_processor, 
            pickled=True,
            ):
        super().__init__()
        if pickled:
            self.env = pickle.loads(env)
            self.env_sampler = pickle.loads(env_sampler)
            self.dynamics_sample_processor = pickle.loads(dynamics_sample_processor)
        else:
            self.env = env
            self.env_sampler = env_sampler
            self.dynamics_sample_processor = dynamics_sample_processor

    def step(self, args=None):
        """
        Uses self.env_sampler which samples data under policy.
        Outcome: generate samples_data.
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
    def __init__(
            self, 
            sample_from_buffer, 
            dynamics_model_max_epochs, 
            dynamics_model, 
            pickled=True,
            ):
        super().__init__()
        self.sample_from_buffer = sample_from_buffer
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        if pickled:
            self.dynamics_model = pickle.loads(dynamics_model)
        else:
            self.dynamics_model = dynamics_model
        self.samples_data = None

    def step(self, args=None):
        '''
        Outcome: dynamics model is updated with self.samples_data.?
        '''

        assert self.samples_data is not None

        time_fit_start = time.time()

        ''' --------------- fit dynamics model --------------- '''

        logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
        self.dynamics_model.fit(self.samples_data['observations'],
                                self.samples_data['actions'],
                                self.samples_data['next_observations'],
                                epochs=self.dynamics_model_max_epochs, verbose=False, log_tabular=True)

        # buffer = None if not self.sample_from_buffer else samples_data

#        logger.record_tabular('Time-ModelFit', time.time() - time_fit_start)

        return pickle.dumps(self.dynamics_model)

    def synch(self, samples_data_pickle):
        self.samples_data = pickle.loads(samples_data_pickle)


class WorkerPolicy(Worker):
    def __init__(
            self, 
            policy, 
            baseline, 
            model_sampler, 
            model_sample_processor, 
            algo, 
            pickled=True,
            ):
        super().__init__()
        if pickled:
            self.policy = pickle.loads(policy)
            self.baseline = pickle.loads(baseline)
            self.model_sampler = pickle.loads(model_sampler)
            self.model_sample_processor = pickle.loads(model_sample_processor)
            self.algo = pickle.loads(algo)
        else:
            self.policy = policy
            self.baseline = baseline
            self.model_sampler = model_sampler
            self.model_sample_processor = model_sample_processor
            self.algo = algo

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
