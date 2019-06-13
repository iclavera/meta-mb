import numpy as np
import time
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
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dicts,
            n_itr,
            start_itr=0,
            steps_per_iter=30,
            initial_random_samples=True,
            dynamics_model_max_epochs=200,
            log_real_performance=True,
            sample_from_buffer=False,
            config=None,
            ):
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.steps_per_iter = steps_per_iter
        self.log_real_performance = log_real_performance
        self.initial_random_samples = initial_random_samples

        worker_instances = [WorkerData(), WorkerModel(dynamics_model_max_epochs), WorkerPolicy()]
        names = ["worker_data", "worker_model", "worker_policy"]
        # command receivers and senders
        receivers, senders = zip(*[Pipe() for _ in range(3)])
        # receipt receivers and senders
        rcp_receivers, rcp_senders = zip(*[Pipe() for _ in range(3)])

        self.ps = [
            Process(
                target=worker_instance,
                name=name,
                args=(
                    policy_pickle,
                    env_pickle,
                    baseline_pickle,
                    dynamics_model_pickle,
                    feed_dict,
                    remote,
                    rcp_sender,
                    synch_notifier,
                    config,
                )
            ) for (worker_instance, name, feed_dict, remote, rcp_sender, synch_notifier) in zip(
                worker_instances, names, feed_dicts, receivers, rcp_senders, senders[1:] + senders[:1]
            )
        ]

        # central scheduler sends command and receives receipts
        self.senders = senders
        self.rcp_receivers = rcp_receivers

    def train(self):
        """
        Trains policy on env using algo
        """
        if type(self.steps_per_iter) is tuple:
            steps_per_iter = np.linspace(self.steps_per_iter[0],
                                            self.steps_per_iter[1], self.n_itr).astype(np.int)
        else:
            steps_per_iter = [self.steps_per_iter] * self.n_itr

        worker_data_sender, worker_model_sender, worker_policy_sender = self.senders
        worker_data_rcp_receiver, worker_model_rcp_receiver, worker_policy_rcp_receiver = self.rcp_receivers

        print("\ntrainer start training...")

        for p in self.ps:
            p.start()

        # initialize worker_model.samples_data
        worker_data_sender.send(('step', self.initial_random_samples))
        # confirm all workes are wamred up
        assert worker_data_rcp_receiver.recv() == 'step done'
        assert worker_data_rcp_receiver.recv() == 'synch done'

        for itr in range(1): #range(self.start_itr, self.n_itr):

            print("\n--------------------starting iteration %s-----------------" % itr)
            
            worker_data_sender.send(('step', None)) 
            worker_model_sender.send(('step', None))

            ''' --------------- MAML steps --------------- '''

            for step in range(steps_per_iter[itr]):

                print("\n--------------------starting step %s-----------------" % step)

                worker_policy_sender.send(('step', None))

        for sender in self.senders:
            sender.send(('close', None))

        for p in self.ps:
            p.join()

        logger.log("Training finished")
