import time
from meta_mb.logger import logger
from multiprocessing import Process, Pipe, Queue, Event
from meta_mb.workers_multi_agents.metrpo.worker_data import WorkerData
from meta_mb.workers_multi_agents.metrpo.worker_model import WorkerModel
from meta_mb.workers_multi_agents.metrpo.worker_policy import WorkerPolicy


NAME_TO_IDX = dict(Data=0, Model=1, Policy=2)


class ParallelTrainerMultiAgents(object):
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
            exp_dir,
            algo_str,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dicts,
            n_itr,
            flags_need_query,
            config,
            num_data_workers,
            num_model_workers,
            num_policy_workers,
            simulation_sleep,
            model_sleep=0,
            policy_sleep=0,
            initial_random_samples=True,
            start_itr=0,
            sampler_str='bptt',
            video=False,
    ):
        self.initial_random_samples = initial_random_samples

        worker_instances = [
            *[WorkerData(time_sleep=simulation_sleep, video=video) for _ in range(num_data_workers)],
            *[WorkerModel(time_sleep=model_sleep) for _ in range(num_model_workers)],
            *[WorkerPolicy(time_sleep=policy_sleep, algo_str=algo_str, sampler_str=sampler_str) for _ in range(num_policy_workers)],
        ]
        names = [
            *[f"Data-{idx}" for idx in range(num_data_workers)],
            *[f"Model-{idx}" for idx in range(num_model_workers)],
            *[f"Policy-{idx}" for idx in range(num_policy_workers)],
        ]
        # one queue for each worker, tasks assigned by scheduler and previous worker
        queues_by_cat = [[Queue(-1) for _ in range(n)] for n in (num_data_workers, num_model_workers, num_policy_workers)]
        queues = sum(queues_by_cat, [])
        name_to_idx = lambda name: NAME_TO_IDX[name.split('-')[0]]
        get_queues_prev = lambda name: queues_by_cat[name_to_idx(name)-1]
        get_queues_next = lambda name: queues_by_cat[(name_to_idx(name)+1)%3]

        # worker sends task-completed notification and time info to scheduler
        worker_remotes, remotes = zip(*[Pipe() for _ in range(num_data_workers+num_model_workers+num_policy_workers)])
        # stop condition
        stop_cond = Event()

        self.ps = [
            Process(
                target=worker_instance,
                name=name,
                args=(
                    exp_dir,
                    policy_pickle,
                    env_pickle,
                    baseline_pickle,
                    dynamics_model_pickle,
                    feed_dicts[name_to_idx(name)],
                    get_queues_prev(name),
                    queue,
                    get_queues_next(name),
                    worker_remote,
                    start_itr,
                    n_itr,
                    stop_cond,
                    False, # need_query,  # all workers don't need query
                    True, # auto_push,  # do auto_push for all workers
                    config,
                ),
            ) for (worker_instance, name, queue, worker_remote) in zip(
                worker_instances, names, queues, worker_remotes
            )
        ]

        # central scheduler sends command and receives receipts
        self.names = names
        self.queues_by_cat = queues_by_cat
        self.remotes = remotes
        self.remotes_by_cat = [remotes[:num_data_workers], remotes[num_data_workers:num_data_workers+num_model_workers], remotes[num_data_workers+num_model_workers:]]

    def train(self):
        """
        Trains policy on env using algo
        """
        worker_data_remotes, worker_model_remotes, worker_policy_remotes = self.remotes_by_cat
        worker_data_queues, worker_model_queues, worker_policy_queues = self.queues_by_cat

        for p in self.ps:
            p.start()

        ''' --------------- worker warm-up --------------- '''

        logger.log('Prepare start...')

        for (remote, queue) in zip(worker_data_remotes, worker_data_queues):
            remote.send('prepare start')
            queue.put(self.initial_random_samples)
            assert remote.recv() == 'loop ready'

        for remote in worker_model_remotes:
            remote.send('prepare start')
            assert remote.recv() == 'loop ready'

        for remote in worker_policy_remotes:
            remote.send('prepare start')
            assert remote.recv() == 'loop ready'

        time_total = time.time()

        ''' --------------- worker looping --------------- '''

        logger.log('Start looping...')
        for remote in self.remotes:
            remote.send('start loop')

        ''' --------------- collect info --------------- '''

        for remote in self.remotes:
            assert remote.recv() == 'loop done'
        logger.log('\n------------all workers exit loops -------------')
        for remote in self.remotes:
            assert remote.recv() == 'worker closed'

        for p in self.ps:
            p.terminate()

        logger.logkv('Trainer-TimeTotal', time.time() - time_total)
        logger.dumpkvs()
        logger.log('*****Training finished')
