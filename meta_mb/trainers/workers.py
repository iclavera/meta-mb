import time, pickle
import numpy as np
from meta_mb.logger import logger
from multiprocessing import current_process
from queue import Empty


class Worker(object):
    """
    Abstract class for worker instantiations. 
    """
    def __init__(
            self, 
            verbose=True,
    ):
        self.verbose = verbose
        self.result = None
        self.state_pickle = None
        self.info = {}

    def __call__(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict,
            queue_prev, 
            queue, 
            queue_next,
            remote,
            start_itr,
            n_itr,
            stop_cond,
            need_query=False, 
            auto_push=True, 
            config=None,
    ):
        """
        Args:
            queue (multiprocessing.Queue): queue for current worker
            queue_next (multiprocessing.Queue): queue for next worker
            rcp_sender (multiprocessing.Connection): notify scheduler after task completed
        """
        self.n_itr = n_itr
        self.queue_prev = queue_prev
        self.queue = queue
        self.queue_next = queue_next

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

            # sess.graph.finalize()

            _init_vars()

            # warm up
            self.itr_counter = start_itr
            assert remote.recv() == 'prepare start'
            self.prepare_start()
            remote.send('loop ready')
            logger.log("\n============== {} is ready =============".format(current_process().name))

            assert remote.recv() == 'start loop'
            time_start = time.time()
            while not stop_cond.is_set():
                if need_query: # poll
                    time_poll = time.time()
                    queue_prev.put('push')
                    self.info.update({current_process().name+'-TimePoll': time.time() - time_poll})
                do_push, do_step = self.process_queue()
                if do_push: # push
                    self.push()
                if do_step: # step
                    self.itr_counter += 1
                    self.step()
                    if auto_push:
                        self.push()

                remote.send(((int(do_push), int(do_step)), self.dump_info()))
                logger.log("\n========================== {} {} {} ===================".format(
                    current_process().name,
                    ('push' if do_push else None, 'step' if do_step else 'synch'),
                    self.itr_counter
                ))
                if self.set_stop_cond():
                    stop_cond.set()

            remote.send('loop done')

            # worker_policy push latest policy
            # worker_data synch and step

            # Alternatively, to avoid repetitive code chunk, let scheduler send latest data
            # FIXME
            """
            data = None
            while True:
                try:
                    data = queue.get_nowait()
                except Empty:
                    break
            assert queue.empty()
            self.prepare_close(data)
            logger.log("\n========== prepared close =====================")
            """
            # remote.send('worker closed')

        logger.logkv(current_process().name+'-TimeTotal', time.time() - time_start)
        logger.dumpkvs()
        logger.log("\n================== {} closed ===================".format(
            current_process().name
        ))

        remote.send('worker closed')

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
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
    
    def process_queue(self):
        push = False
        data = None
        while True:
            try:
                new_data = self.queue.get_nowait()
                if new_data == 'push':
                    push = True
                else:
                    data = new_data
            except Empty:
                break

        # actions is a boolean array
        # actions[0] = push (True) or not (False)
        # actions[1] = step (True) or synch (False)
        actions = [push, data is None]
        if data is not None:
            self._synch(data)
        return actions

    def _synch(self, data):
        raise NotImplementedError

    def dump_result(self):
        self.state_pickle = pickle.dumps(self.result)

    def push(self):
        time_push = time.time()
        self.dump_result()
        self.queue_next.put(self.state_pickle)
        self.info.update({current_process().name+'-TimePush': time.time() - time_push})

    def set_stop_cond(self):
        return False

    def prepare_close(self, args):
        pass

    def update_info(self):
        self.info.update(logger.getkvs())
        logger.reset()

    def dump_info(self):
        info = self.info
        self.info = {}
        return info


class WorkerData(Worker):
    def __init__(self, simulation_sleep):
        super().__init__()
        self.simulation_sleep = simulation_sleep
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
        self.dynamics_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            **feed_dict['dynamics_sample_processor']
        )

    def prepare_start(self):
        initial_random_samples = self.queue.get()
        self.step(initial_random_samples)
        self.queue_next.put(pickle.dumps(self.result))

    def step(self, random=False):
        """
        When args is not None, args = initial_random_samples (bool)
        """

        '''------------- Obtaining samples from the environment -----------'''

        if self.verbose:
            logger.log("Obtaining samples from the environment...")
        time_env_sampling = time.time()
        env_paths = self.env_sampler.obtain_samples(
            log=True,
            random=random,
            log_prefix='Data-EnvSampler-',
        )
        time_env_sampling = time.time() - time_env_sampling

        '''-------------- Processing environment samples -------------------'''

        if self.verbose:
            logger.log("Processing environment samples...")
        # first processing just for logging purposes
        time_env_samp_proc = time.time()
        samples_data = self.dynamics_sample_processor.process_samples(
            env_paths,
            log=True,
            log_prefix='Data-EnvTrajs-'
        )
        time_env_samp_proc = time.time() - time_env_samp_proc

        time.sleep(self.simulation_sleep)
        self.result = samples_data

        # logger.logkv("TimeEnvSampling", time_env_sampling)
        # logger.logkv("TimeEnvSampProc", time_env_samp_proc)
        self.update_info()
        info = {'Data-Iteration': self.itr_counter,
                'Data-TimeEnvSampling': time_env_sampling, 'Data-TimeEnvSampProc': time_env_samp_proc}
        self.info.update(info)

    def _synch(self, policy_state_pickle):
        time_synch = time.time()
        policy_state = pickle.loads(policy_state_pickle)
        assert isinstance(policy_state, dict)
        self.env_sampler.policy.set_shared_params(policy_state)
        time_synch = time.time() - time_synch
        info = {'Data-TimeSynch': time_synch}
        self.info.update(info)

    def set_stop_cond(self):
        return self.itr_counter >= self.n_itr

    def prepare_close(self, data):
        # step one more time with most updated policy to measure performance
        # result dumped in logger
        raise NotImplementedError

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

    def prepare_start(self):
        samples_data_pickle = self.queue.get()
        self._synch(samples_data_pickle)
        self.step()
        self.queue_next.put(pickle.dumps(self.result))

    def step(self):

        assert self.samples_data is not None

        time_model_fit = time.time()

        ''' --------------- fit dynamics model --------------- '''

        if self.verbose:
            logger.log("Training dynamics model for %i epochs ..." % (self.dynamics_model_max_epochs))
        self.dynamics_model.fit(self.samples_data['observations'],
                                self.samples_data['actions'],
                                self.samples_data['next_observations'],
                                epochs=self.dynamics_model_max_epochs, verbose=False,
                                log_tabular=True, prefix='Model-')
        time_model_fit = time.time() - time_model_fit

        self.result = self.dynamics_model

        self.update_info()
        info = {'Model-Iteration': self.itr_counter,
                "Model-TimeModelFit": time_model_fit}
        self.info.update(info)

    def _synch(self, samples_data_pickle):
        time_synch = time.time()
        self.samples_data = pickle.loads(samples_data_pickle)
        time_synch = time.time() - time_synch
        info = {'Model-TimeSynch': time_synch}
        self.info.update(info)

    def dump_result(self):
        self.state_pickle = pickle.dumps(self.result.get_shared_param_values())


class WorkerPolicy(Worker):
    def __init__(self, step_per_iter, warm_next=False):
        super().__init__(warm_next)
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
        time_synch = time.time()
        dynamics_model_state = pickle.loads(dynamics_model_state_pickle)
        assert isinstance(dynamics_model_state, dict)
        self.model_sampler.dynamics_model.set_shared_params(dynamics_model_state)
        self.model_sampler.vec_env.dynamics_model.set_shared_params(dynamics_model_state)
        time_synch = time.time() - time_synch
        info = {'Policy-TimeSynch': time_synch}
        self.info.update(info)

    def dump_result(self):
        self.state_pickle = pickle.dumps(self.result.get_shared_param_values())

    def log_diagnostics(self, paths, prefix):
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
