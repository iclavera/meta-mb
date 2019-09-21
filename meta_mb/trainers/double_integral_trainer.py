import numpy as np
from pdb import set_trace as st
from meta_mb.logger import logger
from meta_mb.replay_buffers import SimpleReplayBuffer
import tensorflow as tf
from meta_mb.utils import create_feed_dict
import time

class Trainer(object):
    def __init__(
            self,
            algo,
            env,
            train_env_sampler,
            eval_env_sampler,
            train_env_sample_processor,
            eval_env_sample_processor,
            dynamics_model,
            policy,
            n_itr,
            sess=None,
            n_initial_exploration_steps=1e3,
            env_max_replay_buffer_size=1e6,
            model_max_replay_buffer_size=2e6,
            rollout_batch_size=100,
            n_train_repeats=1,
            real_ratio=1,
            rollout_length = 1,
            model_deterministic = False,
            model_train_freq=250,
            dynamics_model_max_epochs=50,
            sampler_batch_size=64,
            dynamics_type=0,
            aux_hidden_dim=256,
            T=1,
            ground_truth=False,
            max_epochs_since_update=5,
            num_eval_trajectories=5,
            estimated_iteration = 100,
            ):
        self.algo = algo
        self.env = env
        self.train_env_sampler = train_env_sampler
        self.eval_env_sampler = eval_env_sampler
        self.train_env_sample_processor = train_env_sample_processor
        self.eval_env_sample_processor = eval_env_sample_processor
        self.dynamics_model = dynamics_model
        self.baseline = train_env_sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.n_initial_exploration_steps = n_initial_exploration_steps
        self.env_replay_buffer = SimpleReplayBuffer(self.env, env_max_replay_buffer_size)
        self.model_replay_buffer = SimpleReplayBuffer(self.env, model_max_replay_buffer_size)
        self.rollout_batch_size = rollout_batch_size
        self.rollout_length = rollout_length
        self.n_train_repeats = n_train_repeats
        self.real_ratio = real_ratio
        self.num_eval_trajectories = num_eval_trajectories
        self.model_deterministic = model_deterministic
        self.epoch_length = self.train_env_sampler.max_path_length - 1
        self.model_train_freq = model_train_freq
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.sampler_batch_size = sampler_batch_size
        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.action_dim = int(np.prod(self.env.action_space.shape))
        self.dynamics_type = dynamics_type
        self.T = T
        self.ground_truth = ground_truth
        self.max_epochs_since_update = max_epochs_since_update
        self.estimated_iteration = estimated_iteration

        if sess is None:
            sess = tf.Session()
        self.sess = sess
        self.get_matrices()
        self.build_training_graph()

    def get_matrices(self):
        R = np.eye(self.action_dim)
        Q = np.eye(self.obs_dim)
        A = np.array([[1., self.env.dt], [0., 1.]])
        B = np.array([[0.], [self.env.dt]])
        self.A_array, self.Q_array, self.B_array, self.R_array = A, Q, B, R
        P = Q
        self.p_loss = 1
        itr_count = 0
        while self.p_loss > 0:
            itr_count += 1
            P_new = Q + np.matmul(A.T, np.matmul(P, A)) - np.matmul(A.T,
                      np.matmul(P,
                      np.matmul(B,
                      np.matmul(np.linalg.inv(np.matmul(B.T, np.matmul(P, B)) + R),
                      np.matmul(B.T, np.matmul(P, A))))))
            self.p_loss = ((P - P_new)**2).mean(axis = None)
            P = P_new
        self.K = np.matmul(np.linalg.inv(R + np.matmul(B.T, np.matmul(P, B))), np.matmul(B.T, np.matmul(P, A)))
        logger.log('P converges after ', itr_count, ' iterations.')
        self.P_array = P

    def build_training_graph(self):
        with tf.variable_scope('double_integral_policy', reuse=False):
            self.K = tf.get_variable("K", initializer = self.K)
            self.K = tf.cast(self.K, tf.float32)
            self.P = tf.convert_to_tensor(self.P_array, dtype = tf.float32)
            self.A = tf.convert_to_tensor(self.A_array, dtype = tf.float32)
            self.B = tf.convert_to_tensor(self.B_array, dtype = tf.float32)
            self.Q = tf.convert_to_tensor(self.Q_array, dtype = tf.float32)
            self.R = tf.convert_to_tensor(self.R_array, dtype = tf.float32)

            self.obs_ph = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name='obs')

        self.optimal_Q = self.get_optimal_Q()
        self.estimated_Q = self.get_estimated_Q()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.algo.policy_lr, name="optimizer")
        # optimal_grads = self.optimizer.compute_gradients(self.optimal_Q, var_list=[self.K])[0][0]
        estimated_grads = self.optimizer.compute_gradients(self.estimated_Q, var_list=[self.K])[0][0]
        self.Q_grad_loss = tf.losses.mean_squared_error(optimal_grads, estimated_grads)

    def calculate(self, x, A):
        x = tf.matmul(x, tf.matmul(A, tf.transpose(x)))
        V = tf.reshape(tf.reduce_sum(tf.matmul(tf.identity(x), x), axis = 1), [-1, 1])
        return V

    def optimal_action(self, observations, i = 0):
        return -tf.matmul(observations, tf.transpose(self.K))

    def get_optimal_Q(self):
        best_action = self.optimal_action(self.obs_ph)
        reward = self.env.tf_reward(self.obs_ph, best_action)
        return reward + self.algo.discount * (-0.5) * self.calculate(self.obs_ph, self.P)

    def get_estimated_Q(self):
        result = 0
        obs = self.obs_ph
        for i in range(self.estimated_iteration):
            act = self.optimal_action(obs, i)
            result += (self.algo.discount)**i * (-0.5) * self.calculate(obs, self.Q) + self.calculate(act, self.R)
            obs = tf.transpose(tf.matmul(self.A, tf.transpose(obs)) + tf.matmul(self.B, tf.transpose(act)))
        return result

    def gradient(self, batch):
        observations = batch['observations']

        sess = tf.get_default_session()
        feed_dict = {self.obs_ph: observations}
        return sess.run(self.Q_grad_loss, feed_dict)


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

        saver = tf.train.Saver()
        with self.sess.as_default() as sess:
            sess.run(tf.global_variables_initializer())
            start_time = time.time()

            self.start_itr=0
            self.algo._update_target(tau=1.0)
            while self.env_replay_buffer._size < self.n_initial_exploration_steps:
                paths = self.train_env_sampler.obtain_samples(log=True, log_prefix='train-', random=True)
                samples_data = self.train_env_sample_processor.process_samples(paths, log='all', log_prefix='train-')[0]
                self.env_replay_buffer.add_samples(samples_data['observations'], samples_data['actions'], samples_data['rewards'],
                                                   samples_data['dones'], samples_data['next_observations'])
                self.dynamics_model.update_buffer(samples_data['observations'],
                                           samples_data['actions'],
                                           samples_data['next_observations'])

            time_step = 0
            for itr in range(self.start_itr, self.n_itr):
                self.itr = itr
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")
                paths = self.train_env_sampler.obtain_samples(log=True, log_prefix='train-')
                samples_data = self.train_env_sample_processor.process_samples(paths, log='all', log_prefix='train-')[0]

                fit_start = time.time()
                logger.log("Training models...")
                self.dynamics_model.fit(samples_data['observations'],
                                        samples_data['actions'],
                                        samples_data['next_observations'],
                                        epochs=self.dynamics_model_max_epochs, verbose=False,
                                        log_tabular=True, prefix='Model-',
                                        max_epochs_since_update=self.max_epochs_since_update)
                logger.logkv('Fit model time', time.time() - fit_start)
                logger.log("Done training models...")
                expand_model_replay_buffer_time = []
                sac_time = []

                for _ in range(self.epoch_length // self.model_train_freq):
                    expand_model_replay_buffer_start = time.time()
                    samples_num = int(self.rollout_batch_size)
                    random_states = self.env_replay_buffer.random_batch_simple(samples_num)['observations']
                    actions_from_policy = self.policy.get_actions(random_states)[0]
                    next_obs, rewards, term = self.step(random_states, actions_from_policy)
                    self.model_replay_buffer.add_samples(random_states,
                                                         actions_from_policy,
                                                         rewards,
                                                         term,
                                                         next_obs)
                    expand_model_replay_buffer_time.append(time.time() - expand_model_replay_buffer_start)

                    sac_start = time.time()

                    for _ in range(self.model_train_freq * self.n_train_repeats):
                        batch_size = self.sampler_batch_size
                        env_batch_size = int(batch_size * self.real_ratio)
                        model_batch_size = batch_size - env_batch_size
                        env_batch = self.env_replay_buffer.random_batch(env_batch_size)
                        model_batch = self.model_replay_buffer.random_batch(int(model_batch_size))
                        keys = env_batch.keys()
                        batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
                        Q_grad_loss = self.gradient(batch)
                        # self.algo.do_training(time_step, batch, log=True)

                    logger.logkv('Q_grad_loss', Q_grad_loss)

                    sac_time.append(time.time() - sac_start)
                self.env_replay_buffer.add_samples(samples_data['observations'],
                                                   samples_data['actions'],
                                                   samples_data['rewards'],
                                                   samples_data['dones'],
                                                   samples_data['next_observations'])


                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.train_env_sampler.total_timesteps_sampled)
                logger.logkv('ItrTime', time.time() - itr_start_time)
                logger.logkv('SAC Training Time', sum(sac_time))
                logger.logkv('Model Rollout Time', sum(expand_model_replay_buffer_time))

                logger.log("Saving snapshot...")

                if itr == 0:
                    sess.graph.finalize()
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.dumpkvs()
                logger.log("Saved")


        logger.log("Training finished")
        self.sess.close()


    def step(self, obs, actions):
        assert self.dynamics_type in [0, 3]
        if self.dynamics_type == 0 or self.dynamics_type == 3:
            next_observation = self.dynamics_model.predict(obs, actions)
            rewards = self.env.reward(obs, actions, next_observation)
            dones = self.env.termination_fn(obs, actions, next_observation)

        dones = dones.reshape((-1))
        rewards = rewards.reshape((-1))
        return next_observation, rewards, dones

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr,
                    policy=self.policy,
                    env=self.env,
                    baseline=self.baseline,
                    dynamics=self.dynamics_model,
                    vfun=self.algo.Qs,
                )
