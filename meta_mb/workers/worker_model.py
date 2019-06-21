import time, pickle
from meta_mb.logger import logger
from meta_mb.workers.base import Worker


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
        # time_synch = time.time()
        self.samples_data = pickle.loads(samples_data_pickle)
        #time_synch = time.time() - time_synch
        #info = {'Model-TimeSynch': time_synch}
        #self.info.update(info)

    def dump_result(self):
        self.state_pickle = pickle.dumps(self.result.get_shared_param_values())

