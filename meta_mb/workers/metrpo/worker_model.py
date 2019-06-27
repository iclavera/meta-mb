import time, pickle
from meta_mb.logger import logger
from meta_mb.workers.base import Worker
from queue import Empty


class WorkerModel(Worker):
    def __init__(self):
        super().__init__()
        self.with_new_data = None
        self.remaining_model_idx = None
        self.valid_loss_rolling_average = None
        self.dynamics_model = None

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
        self._synch(samples_data_pickle, check_init=True)
        self.step()
        self.queue_next.put(pickle.dumps(self.result))

    def process_queue(self):
        do_push, do_synch = 0, 0

        while True:
            try:
                if not self.remaining_model_idx:
                    logger.log('Model at iteration {} is block waiting for data'.format(self.itr_counter))
                    new_data = self.queue.get()
                else:
                    if self.verbose:
                        logger.log('Model try get_nowait.........')
                    new_data = self.queue.get_nowait()
                if new_data == 'push':
                    # Only push once before executing another step
                    if do_push == 0:
                        do_push = 1
                        self.push()
                else:
                    do_synch += 1
                    self._synch(new_data)
            except Empty:
                if self.verbose:
                    logger.log('Model queue Empty')
                break

        do_step = 1 - do_synch

        if self.verbose:
            logger.log('Model finishes processing queue with {}, {}, {}......'.format(do_push, do_synch, do_step))

        return do_push, do_synch, do_step

    def step(self, obs=None, act=None, obs_next=None):

        """ --------------- fit dynamics model --------------- """

        time_model_fit = time.time()
        if self.verbose:
            logger.log('Model at iteration {} is training for one epoch...'.format(self.itr_counter))
        self.remaining_model_idx, self.valid_loss_rolling_average = self.dynamics_model.fit_one_epoch(
            remaining_model_idx=self.remaining_model_idx,
            valid_loss_rolling_average_prev=self.valid_loss_rolling_average,
            with_new_data=self.with_new_data,
            verbose=self.verbose,
            log_tabular=True,
            prefix='Model-',
        )
        time_model_fit = time.time() - time_model_fit

        self.result = self.dynamics_model
        self.with_new_data = False

        info = {'Model-Iteration': self.itr_counter,
                "Model-TimeModelFit": time_model_fit}
        logger.logkvs(info)

    def _synch(self, samples_data_pickle, check_init=False):
        if self.verbose:
            logger.log('Model at {} is synchronizing...'.format(self.itr_counter))
        time_synch = time.time()
        samples_data = pickle.loads(samples_data_pickle)
        self.dynamics_model.update_buffer(
            samples_data['observations'],
            samples_data['actions'],
            samples_data['next_observations'],
            check_init=check_init,
        )

        # Reset variables for early stopping condition
        self.with_new_data = True
        self.remaining_model_idx = list(range(self.dynamics_model.num_models))
        self.valid_loss_rolling_average = None
        time_synch = time.time() - time_synch

        logger.logkv('Model-TimeSynch', time_synch)

    def dump_result(self):
        self.state_pickle = pickle.dumps(self.result.get_shared_param_values())

    def push(self):
        time_push = time.time()
        self.dump_result()
        # FIXME: only works when all workers do autopush, because it removes "push" message.
        #  To support autopush = False, set two different queues for data and "push" message.
        #  5 is arbitrary.
        while self.queue_next.qsize() > 3:
            try:
                logger.log('Model is off loading data from queue_next...')
                self.queue_next.get_nowait()
            except Empty:
                break
        self.queue_next.put(self.state_pickle)
        time_push = time.time() - time_push
        logger.logkv('Model-TimePush', time_push)

