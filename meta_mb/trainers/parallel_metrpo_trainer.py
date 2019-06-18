from collections import defaultdict
import time
from meta_mb.logger import logger
from multiprocessing import Process, Pipe, Queue, Event
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
            flags_need_query=(True, False, True),
            config=None,
            ):
        self.log_real_performance = log_real_performance
        self.initial_random_samples = initial_random_samples
        self.summary = defaultdict(list)

        if type(steps_per_iter) is tuple:
            step_per_iter = int((steps_per_iter[1] + steps_per_iter[0])/2)
        else:
            step_per_iter = steps_per_iter
        assert step_per_iter > 0

        worker_instances = [WorkerData(), WorkerModel(dynamics_model_max_epochs), WorkerPolicy(step_per_iter)]
        names = ["Data", "Model", "Policy"]
        # one queue for each worker, tasks assigned by scheduler and previous worker
        queues = [Queue() for _ in range(3)]
        # worker sends task-completed notification and time info to scheduler
        worker_remotes, remotes = zip(*[Pipe() for _ in range(3)])
        # stop condition
        stop_cond = Event()
        # current worker needs query means previous workers does not auto push
        # skipped checking here
        flags_need_query = flags_need_query
        flags_auto_push = [not flags_need_query[1], not flags_need_query[2], not flags_need_query[0]]

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
                    queue_prev,
                    queue,
                    queue_next,
                    worker_remote,
                    start_itr,
                    n_itr,
                    stop_cond,
                    need_query,
                    auto_push,
                    config,
                ),
            ) for (worker_instance, name, feed_dict,
                   queue_prev, queue, queue_next,
                   worker_remote, need_query, auto_push) in zip(
                worker_instances, names, feed_dicts,
                queues[2:] + queues[:2], queues, queues[1:] + queues[:1],
                worker_remotes, flags_need_query, flags_auto_push,
            )
        ]

        # central scheduler sends command and receives receipts
        self.names = names
        self.queues = queues
        self.remotes = remotes

    def train(self):
        """
        Trains policy on env using algo
        """
        worker_data_queue, worker_model_queue, worker_policy_queue = self.queues
        worker_data_remote, worker_model_remote, worker_policy_remote = self.remotes

        print("\ntrainer start training...")

        for p in self.ps:
            p.start()

        ''' --------------- worker warm-up --------------- '''

        logger.log('Prepare start...')

        worker_data_remote.send('prepare start')
        worker_data_queue.put(self.initial_random_samples)
        msg = worker_data_remote.recv()
        assert msg == 'loop ready'

        worker_model_remote.send('prepare start')
        msg = worker_model_remote.recv()
        assert msg == 'loop ready'

        worker_policy_remote.send('prepare start')
        msg = worker_policy_remote.recv()
        assert msg == 'loop ready'

        time_total = time.time()

        ''' --------------- worker looping --------------- '''

        logger.log('Start looping...')
        for remote in self.remotes:
            remote.send('start loop')

        ''' --------------- collect info --------------- '''

        self.collect_summary('loop done')
        logger.log('\n------------all workers exit loops -------------')
        self.collect_summary('worker closed')
        logger.logkv('Trainer-TimeTotal', time.time() - time_total)

        for key, value in self.summary.items():
            print(key)
            value = value[:-2]
            do_push, do_step = zip(*value)
            print('do_push', do_push)
            print('do_step', do_step)
            logger.logkv('{}-PushPercentage'.format(key), sum(do_push)/len(value))
            logger.logkv('{}-StepPercentage'.format(key), sum(do_step)/len(value))
        logger.log("Training finished")
        logger.dumpkvs()

    def collect_summary(self, stop_rcp):
        for name, remote in zip(self.names, self.remotes):
            tasks = []
            while True:
                rcp = remote.recv()
                if rcp == stop_rcp:
                    print('receiving stop rcp {}'.format(rcp))
                    break
                task, info = rcp
                tasks.append(task)
                assert isinstance(info, dict)
                logger.logkvs(info)
                logger.dumpkvs()
            self.summary[name].extend(tasks)
