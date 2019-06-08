import tensorflow as tf
import numpy as np
import time
from meta_mb.logger import logger
from multiprocessing import Process, Pipe

class ParallelTrainer(object):
    """
    Performs steps for MAML

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            algo,
            env,
            model_sampler,
            env_sampler,
            model_sample_processor,
            dynamics_sample_processor,
            policy,
            dynamics_model,
            n_itr,
            start_itr=0,
            steps_per_iter=30,
            initial_random_samples=True,
            sess=None,
            dynamics_model_max_epochs=200,
            log_real_performance=True,
            sample_from_buffer=False,
            ):
#        self.algo = algo
#        self.env = env
#        self.model_sampler = model_sampler
#        self.model_sample_processor = model_sample_processor
#        self.baseline = model_sample_processor.baseline
#        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
#
        self.steps_per_iter = steps_per_iter
#        self.log_real_performance = log_real_performance

        if sess is None:
            sess = tf.Session()
        self.sess = sess

        self.worker_instances = [
                WorkerData(initial_random_samples, env_sampler, dynamics_sample_processor), 
                WorkerModel(sample_from_buffer, dynamics_model_max_epochs, dynamics_model), 
                WorkerPolicy(),
                ]

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(3)])
        seeds = np.random.choice(range(10**6), size=3, replace=False)

        self.ps = [
                Process(target=worker_instance, args=(remote, seed))
                for (worker_instance, remote, seed) in zip(
                    self.worker_instances, self.remotes, seeds)]

        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def train(self):
        """
        Trains policy on env using algo

        Pseudocode:
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """
        worker_data, worker_model, worker_policy = self.worker_instances

        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            if type(self.steps_per_iter) is tuple:
                steps_per_iter = np.linspace(self.steps_per_iter[0],
                                             self.steps_per_iter[1], self.n_itr).astype(np.int)
            else:
                steps_per_iter = [self.steps_per_iter] * self.n_itr

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                self.worker_data.send(('step', None)) 

                self.worker_model.send(('step', None))

                ''' --------------- MAML steps --------------- '''
                times_dyn_sampling = []
                times_dyn_sample_processing = []
                times_optimization = []
                times_step = []

                for step in range(steps_per_iter[itr]):

                    logger.log("\n ---------------- Grad-Step %d ----------------" % int(sum(steps_per_iter[:itr]) + step))

                    self.worker_model.send(('step', None))



