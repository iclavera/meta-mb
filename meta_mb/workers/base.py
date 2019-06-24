import time, pickle
from meta_mb.logger import logger
from multiprocessing import current_process
from queue import Empty


class Worker(object):
    """
    Abstract class for worker instantiations. 
    """
    def __init__(
            self,
            verbose=False,
    ):
        """
        :param
            max_num_data_dropped (int): maximum number of data dropped when processing task queue
        """

        self.verbose = verbose
        self.result = None
        self.state_pickle = None
        self.info = {}

    def __call__(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict,
            queue_prev, 
            queue, 
            queue_next,
            remote,
            start_itr,
            n_itr,
            stop_cond,
            need_query=False,
            auto_push=True,
            config=None,
    ):
        """
        Args:
            queue (multiprocessing.Queue): queue for current worker
            queue_next (multiprocessing.Queue): queue for next worker
            rcp_sender (multiprocessing.Connection): notify scheduler after task completed
        """
        self.n_itr = n_itr
        self.queue_prev = queue_prev
        self.queue = queue
        self.queue_next = queue_next
        self.stop_cond = stop_cond

        import tensorflow as tf

        def _init_vars():
            sess = tf.get_default_session()
            # uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            # sess.run(tf.variables_initializer(uninit_vars))
            sess.run(tf.initializers.global_variables())

        with tf.Session(config=config).as_default() as sess:

            self.construct_from_feed_dict(
                policy_pickle,
                env_pickle,
                baseline_pickle,
                dynamics_model_pickle,
                feed_dict,
            )

            # sess.graph.finalize()

            _init_vars()

            # warm up
            # TODO: Add warm up rounds
            self.itr_counter = start_itr
            if self.verbose:
                print('working waiting for starting msg from trainer...')
            assert remote.recv() == 'prepare start'
            self.prepare_start()
            remote.send('loop ready')
            logger.log("\n============== {} is ready =============".format(current_process().name))

            assert remote.recv() == 'start loop'
            time_start = time.time()
            while not self.stop_cond.is_set():
                if self.verbose:
                    logger.log("\n------------------------- {} starting new loop ------------------".format(current_process().name))
                if need_query: # poll
                    # time_poll = time.time()
                    queue_prev.put('push')
                    # self.info.update({current_process().name+'-TimePoll': time.time() - time_poll})
                do_push, do_synch, do_step= self.process_queue()
                # step
                if do_step:
                    self.itr_counter += 1
                    self.step()
                    if auto_push:
                        self.push()

                remote.send(((int(do_push), int(do_synch), int(do_step)), self.dump_info()))
                # remote.send(((int(do_push), int(do_synch), int(do_step)), None))
                # logger.logkvs(self.dump_info())
                # logger.dumpkvs()
                logger.log("\n========================== {} {} {} ===================".format(
                    current_process().name,
                    ('push' if do_push else None, do_synch, 'step' if do_step else None),
                    self.itr_counter
                ))
                self.set_stop_cond()

            remote.send('loop done')

            # TODO: evaluate performance with most recent policy?
            # worker_policy push latest policy
            # worker_data synch and step
            # Alternatively, to avoid repetitive code chunk, let scheduler send latest data
            """
            data = None
            while True:
                try:
                    data = queue.get_nowait()
                except Empty:
                    break
            assert queue.empty()
            self.prepare_close(data)
            logger.log("\n========== prepared close =====================")
            """
            # remote.send('worker closed')

        logger.logkv(current_process().name+'-TimeTotal', time.time() - time_start)
        logger.dumpkvs()
        logger.log("\n================== {} closed ===================".format(
            current_process().name
        ))

        remote.send('worker closed')

    def construct_from_feed_dict(
            self,
            policy_pickle,
            env_pickle,
            baseline_pickle,
            dynamics_model_pickle,
            feed_dict
    ):
        raise NotImplementedError

    def prepare_start(self):
        raise NotImplementedError

    def process_queue(self):
        if self.verbose:
            logger.log('{} start processing queu.................'.format(current_process().name))
        do_push, do_synch = False, 0
        data = None
        while True:
            try:
                new_data = self.queue.get_nowait()
                if new_data == 'push':
                    if not do_push: # only push once
                        do_push = True
                        self.push()
                else:
                    do_synch += 1
                    data = new_data
            except Empty:
                break

        if do_synch:
            self._synch(data)

        do_step = not do_synch

        if self.verbose:
            logger.log('{} finishes processing queue with {}, {}, {}......'.format(current_process().name, do_push, do_synch, do_step))
        return do_push, do_synch, do_step

    def step(self):
        raise NotImplementedError

    def _synch(self, data):
        raise NotImplementedError

    def dump_result(self):
        self.state_pickle = pickle.dumps(self.result)

    def push(self):
        # time_push = time.time()
        self.dump_result()
        self.queue_next.put(self.state_pickle)
        # self.info.update({current_process().name+'-TimePush': time.time() - time_push})

    def set_stop_cond(self):
        pass

    def prepare_close(self, args):
        pass

    def update_info(self):
        self.info.update(logger.getkvs())
        logger.reset()

    def dump_info(self):
        info = self.info
        self.info = {}
        return info

