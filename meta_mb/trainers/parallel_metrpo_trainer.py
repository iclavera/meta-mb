import numpy as np
import time
from meta_mb.logger import logger
from multiprocessing import Process, Pipe, Queue
from meta_mb.trainers.workers import WorkerData, WorkerModel, WorkerPolicy

TIMEOUT = 1000


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
        # one queue for each worker, tasks assigned by scheduler and previous worker
        queues = [Queue() for _ in range(3)]
        # worker sends task-completed notification and time info to scheduler
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
                    queue,
                    queue_next,
                    rcp_sender,
                    start_itr,
                    config,
                )
            ) for (worker_instance, name, feed_dict, queue, rcp_sender, queue_next) in zip(
                worker_instances, names, feed_dicts, queues, rcp_senders, queues[1:] + queues[:1]
            )
        ]

        # central scheduler sends command and receives receipts
        self.names = names
        self.queues = queues
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

        worker_data_queue, worker_model_queue, worker_policy_queue = self.queues
        worker_data_rcp_receiver, worker_model_rcp_receiver, worker_policy_rcp_receiver = self.rcp_receivers

        print("\ntrainer start training...")

        for p in self.ps:
            p.start()

        # warm up all workers in chain
        worker_data_queue.put(('step', self.initial_random_samples), block=True, timeout=TIMEOUT)
        # confirm all workers are warmed up
        assert worker_data_rcp_receiver.recv() == 'step {} done'.format(self.start_itr-1)
        assert worker_data_rcp_receiver.recv() == 'synch done'

        time_total = time.time()
        # put tasks onto queues
        for itr in range(self.start_itr, self.n_itr):
            worker_data_queue.put(('step', None), block=True, timeout=TIMEOUT)
            worker_model_queue.put(('step', None), block=True, timeout=TIMEOUT)
            worker_policy_queue.put(('step', steps_per_iter[itr]), block=True, timeout=TIMEOUT)
            assert worker_data_rcp_receiver.recv() == 'step {} done'.format(itr)
            assert worker_data_rcp_receiver.recv() == 'synch done'

        for queue in self.queues:
            queue.put(('close', None))

        # collect timing info
        logger.log("Logging summary...")
        summary = {}
        for name, rcp_receiver in zip(self.names, self.rcp_receivers):
            tasks = []
            while True:
                rcp = rcp_receiver.recv()
                tasks.append(rcp)
                if rcp == 'worker exists':
                    # info = rcp_receiver.recv()
                    break
            print("\n-------------------------------------\n")
            print(name)
            print(tasks)
            summary[name] = (tasks)

        logger.logkv('time_total', time.time() - time_total)
        logger.dumpkvs()

        logger.log("Training finished")

