import time, pickle
from meta_mb.logger import logger
from meta_mb.workers.base import Worker
from queue import Empty


class WorkerModelBase(Worker):
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
            feed_dict,
    ):
        self.dynamics_model = pickle.loads(dynamics_model_pickle)

    def prepare_start(self):
        self._synch(self.queue.get(), check_init=True)
        self.step()
        self.queue_next.put(pickle.dumps(self.dynamics_model))

    def process_queue(self):
        do_push, do_synch = 0, 0

        while True:
            try:
                if not self.remaining_model_idx:
                    new_data = self.queue.get(block=True, timeout=100)
                else:
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
                break

        if self.remaining_model_idx:
            do_step = 1
        else:  # early stopped
            do_step = 0

        return do_push, do_synch, do_step

    def step(self):

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

        self.with_new_data = False

        logger.logkv('Model-TimeStep', time_model_fit)

    def _synch(self, samples_data_pickle, check_init=False):

        if self.verbose:
            logger.log('Model at {} is synchronizing...'.format(self.itr_counter))

        time_synch = time.time()
        act, obs, obs_next = pickle.loads(samples_data_pickle)
        self.dynamics_model.update_buffer(
            act=act,
            obs=obs,
            obs_next=obs_next,
            check_init=check_init,
        )
        # Reset variables for early stopping condition
        self.with_new_data = True
        self.remaining_model_idx = list(range(self.dynamics_model.num_models))
        self.valid_loss_rolling_average = None
        time_synch = time.time() - time_synch

        logger.logkv('Model-TimeSynch', time_synch)

    def push(self):
        time_push = time.time()
        state_pickle = pickle.dumps(self.dynamics_model.get_shared_param_values())
        while self.queue_next.qsize() > 3:
            try:
                logger.log('Model is off loading data from queue_next...')
                _ = self.queue_next.get_nowait()
            except Empty:
                break
        assert state_pickle is not None
        self.queue_next.put(state_pickle)
        time_push = time.time() - time_push

        logger.logkv('Model-TimePush', time_push)

    """
    def push(self):
        time_push = time.time()
        self.dump_result()
        put_msg_push = False # TODO: havn't tested with any(flags_need_query) is True
        while self.queue_next.qsize() > 3:
            try:
                logger.log('Model is off loading data from queue_next...')
                data = self.queue_next.get_nowait()
                if data == 'push':
                    put_msg_push = True
            except Empty:
                break
        self.queue_next.put(self.state_pickle)
        if put_msg_push:
            self.queue_next.put('push')
        time_push = time.time() - time_push

        logger.logkv('Model-TimePush', time_push)
    """