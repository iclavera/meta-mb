import time, pickle
from meta_mb.logger import logger
import multiprocessing

TIMEOUT = 100


class Worker(object):
    """
    Abstract class for worker instantiations. 
    """
    def __init__(self, warm_next):
        """
        Args:
            warm_next (bool): whether to synch and step next worker at the very start
        """
        self.warm_next = warm_next

    def __call__(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict,
            queue,
            queue_next,
            rcp_sender,
            config=None,
    ):
        """
        Args:
            remote (multiprocessing.Connection):
            queue_next (multiprocessing.Connection):
        """
        import tensorflow as tf

        def _init_vars():
            sess = tf.get_default_session()
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

        with tf.Session(config=config).as_default() as sess:

            self.construct_from_feed_dict(
                policy_pickle,
                env_pickle,
                baseline_pickle,
                dynamics_model_pickle,
                feed_dict,
            )

            _init_vars()

            while True:
                cmd, args = queue.get()

                if cmd == 'step':
                    result_pickle = self.step(args)
                    queue_next.put(('synch', result_pickle), block=True, timeout=TIMEOUT)
                    if self.warm_next:
                        queue_next.put(('step', None))
                        self.warm_next = False

                elif cmd == 'synch':
                    self.synch(args)

                elif cmd == 'start_chain':
                    result_pickle = self.step(args)
                    queue_next.put(('synch_and_chain', result_pickle), block=True, timeout=TIMEOUT)

                elif cmd == 'synch_and_chain':
                    self.synch(args)
                    result_pickle = self.step(args)
                    queue_next.put(('synch_and_chain', result_pickle), block=True, timeout=TIMEOUT)

                elif cmd == 'close':
                    break

                else:
                    raise NotImplementedError

                # Notify trainer that one more task is completed
                rcp_sender.send('{} done'.format(cmd))
                print("\n-----------------" + multiprocessing.current_process().name + " finishing " + cmd)
        sess.close()
        print("\n---------------{} hitting exit".format(multiprocessing.current_process().name))
        rcp_sender.send('worker exists')

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
    def __init__(self, warm_next=True):
        super().__init__(warm_next)
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
    def __init__(self, dynamics_model_max_epochs, warm_next=True):
        super().__init__(warm_next)
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
    def __init__(self, warm_next=False):
        super().__init__(warm_next)
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
        dynamics_model = pickle.loads(dynamics_model_pickle)
        self.model_sampler.dynamics_model = dynamics_model
        self.model_sampler.vec_env.dynamics_model = dynamics_model
        assert dynamics_model.normalization is not None

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
