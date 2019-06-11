import tensorflow as tf
import numpy as np
import time, pickle
from meta_mb.logger import logger
from multiprocessing import Process, Pipe
from meta_mb.trainers.workers import WorkerData, WorkerModel, WorkerPolicy

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
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.steps_per_iter = steps_per_iter
        self.log_real_performance = log_real_performance

        if sess is None:
            sess = tf.Session()
        self.sess = sess

        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

        print("\n--------------------------------dump test starts")
        pickle.dumps(env_sampler)
        print("\n--------------------------------dump test passed")

        self.worker_instances = [
                WorkerData(
                    initial_random_samples, 
                    pickle.dumps(env), 
                    pickle.dumps(env_sampler), 
                    pickle.dumps(dynamics_sample_processor), 
                    ), 
                WorkerModel(
                    sample_from_buffer, 
                    dynamics_model_max_epochs, 
                    pickle.dumps(dynamics_model), 
                    ), 
                WorkerPolicy(
                    pickle.dumps(policy), 
                    pickle.dumps(model_sample_processor.baseline), 
                    pickle.dumps(model_sampler), 
                    pickle.dumps(model_sample_processor), 
                    pickle.dumps(algo)),
                ]

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
        
        names = ["worker_data", "worker_model", "worker_policy"]
        receivers, senders = zip(*[Pipe() for _ in range(3)])
        worker_data_sender, worker_model_sender, worker_policy_sender = senders
        worker_data, worker_model, worker_policy = self.worker_instances

        self.ps = [
                Process(target=worker_instance, name=name, args=(remote, synch_notifier))
                for (worker_instance, name, remote, synch_notifier) in zip(
                    self.worker_instances, names, receivers, 
                    [worker_model_sender, worker_policy_sender, worker_data_sender])]

        # TODO: close?
        
        print("\ntrainer start training...")

        with self.sess.as_default() as sess:

            if type(self.steps_per_iter) is tuple:
                steps_per_iter = np.linspace(self.steps_per_iter[0],
                                             self.steps_per_iter[1], self.n_itr).astype(np.int)
            else:
                steps_per_iter = [self.steps_per_iter] * self.n_itr

    
            # initialize worker_model.samples_data
            samples_data = worker_data.init_step()
            worker_model.synch(pickle.dumps(samples_data))

            self.ps[0].start()

            for itr in range(5): #range(self.start_itr, self.n_itr):

                print("\n--------------------starting iteration %s-----------------" % itr)
                
                worker_data_sender.send(('step', None)) 
#                worker_data.step()

                # worker_model_sender.send(('step', None))

                ''' --------------- MAML steps --------------- '''

#                for step in range(steps_per_iter[itr]):

                # worker_policy_sender.send(('step', None))

                #for p in self.ps:
                #    p.join()

        for sender in senders:
            sender.close()

        self.ps[0].join()
        # for p in self.ps:
        #    p.terminate()
        #    p.join()

        logger.log("Training finished")
        self.sess.close()
