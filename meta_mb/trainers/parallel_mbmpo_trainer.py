from collections import defaultdict
import time
from multiprocessing import Process, Pipe, Queue, Event
from meta_mb.logger import logger
from meta_mb.workers.mbmpo.worker_data import WorkerData
from meta_mb.workers.mbmpo.worker_model import WorkerModel
from meta_mb.workers.mbmpo.worker_policy import WorkerPolicy


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
            num_inner_grad_steps=1,
            meta_steps_per_iter=30,
            initial_random_samples=True,
            dynamics_model_max_epochs=200,
            log_real_performance=True,
            flags_need_query=(False, False, False,),
            sample_from_buffer=False,
            fraction_meta_batch_size=1,
            config=None,
            simulation_sleep=10,
            ):

        self.log_real_performance = log_real_performance
        self.initial_random_samples = initial_random_samples
        self.summary = defaultdict(list)

        if type(meta_steps_per_iter) is tuple:
            meta_step_per_iter = int((meta_steps_per_iter[1] + meta_steps_per_iter[0])/2)
        else:
            meta_step_per_iter = meta_steps_per_iter

        worker_instances = [
            WorkerData(fraction_meta_batch_size, simulation_sleep),
            WorkerModel(dynamics_model_max_epochs),
            WorkerPolicy(sample_from_buffer, meta_step_per_iter, num_inner_grad_steps),
        ]
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
        [p.terminate() for p in self.ps]
        logger.logkv('Trainer-TimeTotal', time.time() - time_total)

        for key, value in self.summary.items():
            print(key)
            value = value[:-2]
            do_push, do_synch, do_step = zip(*value)
            logger.logkv('{}-PushPercentage'.format(key), sum(do_push)/len(value))
            logger.logkv('{}-SynchPercentage'.format(key), sum(do_synch)/len(value))
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
                if info:
                    logger.logkvs(info)
                    logger.dumpkvs()
            self.summary[name].extend(tasks)

