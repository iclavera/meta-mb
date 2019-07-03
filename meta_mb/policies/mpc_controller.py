from meta_mb.utils.serializable import Serializable
import tensorflow as tf
import numpy as np


class MPCController(Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            reward_model=None,
            discount=1,
            use_cem=False,
            n_candidates=1024,
            horizon=10,
            num_cem_iters=8,
            percent_elites=0.1,
            use_reward_model=False,
            alpha=0.1,
            num_particles=20,
    ):
        Serializable.quick_init(self, locals())
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.use_cem = use_cem
        self.num_cem_iters = num_cem_iters
        self.percent_elites = percent_elites
        self.env = env
        self.use_reward_model = use_reward_model
        self.alpha = alpha
        self.num_particles = num_particles

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1
        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        if not use_cem:
            self.build_rs_graph()

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]

        if self.use_cem:
            action = self.get_cem_action(observation)
        else:
            action = self.get_rs_action(observation)

        return action, dict()

    def get_actions(self, observations):
        if self.use_cem:
            actions = self.get_cem_action(observations)
        else:
            actions = self.get_rs_action(observations)

        #for i in range(3, len(actions)): #limit movement to first 3 joints
        #    actions[i] = 0
        return actions, dict()

    def get_random_action(self, n):
        return np.random.uniform(low=self.env.action_space.low,
                                 high=self.env.action_space.high, size=(n,) + self.env.action_space.low.shape)

    def get_sinusoid_actions(self, action_space, t):
        actions = np.array([])
        delta = t/action_space
        for i in range(action_space):
            #actions = np.append(actions, 0.5 * np.sin(i * delta)) #two different ways of sinusoidal sampling
            actions = np.append(actions, 0.5 * np.sin(i * t))
        #for i in range(3, len(actions)): #limit movement to first 3 joints
        #    actions[i] = 0
        return actions

    def get_cem_action(self, observations):

        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        act_dim = self.env.action_space.shape[0]

        num_elites = max(int(self.n_candidates * self.percent_elites), 1)
        mean = np.ones((m, h * act_dim)) * (self.env.action_space.high + self.env.action_space.low)/2
        std = np.ones((m, h * act_dim)) * (self.env.action_space.high - self.env.action_space.low)/16
        clip_low = np.concatenate([self.env.action_space.low] * h)
        clip_high = np.concatenate([self.env.action_space.high] * h)

        for i in range(self.num_cem_iters):
            z = np.random.normal(size=(n, m,  h * act_dim))
            a = mean + z * std
            a = np.clip(a, clip_low, clip_high)
            a_stacked = a.copy()
            a = a.reshape((n * m, h, act_dim))
            a = np.transpose(a, (1, 0, 2))
            returns = np.zeros((n * m * self.num_particles,))

            cand_a = a[0].reshape((m, n, -1))
            observation = np.repeat(observations, n * self.num_particles, axis=0)
            for t in range(h):
                a_t = np.repeat(a[t], self.num_particles, axis=0)
                next_observation = self.dynamics_model.predict(observation, a_t, deterministic=False)
                rewards = self.unwrapped_env.reward(observation, a_t, next_observation)
                returns += self.discount ** t * rewards
                observation = next_observation
            returns = np.mean(np.split(returns.reshape(m, n * self.num_particles),
                                       self.num_particles, axis=-1), axis=0)   # TODO: Make sure this reshaping works
            elites_idx = ((-returns).argsort(axis=-1) < num_elites).T
            elites = a_stacked[elites_idx]
            mean = mean * self.alpha + (1 - self.alpha) * np.mean(elites, axis=0)
            std = np.std(elites, axis=0)
            lb_dist, ub_dist = mean - self.env.action_space.low, self.env.action_space.high - mean
            std = np.minimum(np.minimum(lb_dist/2, ub_dist/2), std)

        return cand_a[range(m), np.argmax(returns, axis=1)]

    def build_rs_graph(self):
        # FIXME: not sure if it workers for batch_size > 1 (num_rollouts > 1)
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_space_dims), name='obs')
        returns = 0  # (batch_size * n_candidates,)
        act = tf.random.uniform(
            shape=[self.horizon, tf.shape(self.obs_ph)[0] * self.n_candidates, self.action_space_dims],
            minval=self.env.action_space.low,
            maxval=self.env.action_space.high)

        # Equivalent to np.repeat
        observation = tf.reshape(
            tf.tile(tf.expand_dims(self.obs_ph, -1), [1, self.n_candidates, 1]),
            [-1, self.obs_space_dims]
        )
        # observation = tf.concat([self.obs_ph for _ in range(self.n_candidates)], axis=0)

        for t in range(self.horizon):
            # dynamics_dist = self.dynamics_model.distribution_info_sym(observation, act[t])
            # mean, var = dynamics_dist['mean'], dynamics_dist['var']
            # next_observation = mean + tf.random.normal(shape=tf.shape(mean))*tf.sqrt(var)
            next_observation = self.dynamics_model.predict_sym(observation, act[t])
            assert self.reward_model is None
            rewards = self.unwrapped_env.tf_reward(observation, act[t], next_observation)
            returns = self.discount ** t * rewards
            observation = next_observation
        """
        returns = tf.reshape(returns, (self.n_candidates, -1))
        idx = tf.reshape(tf.argmax(returns, axis=0), [-1, 1])  # (batch_size, 1)
        cand_a = tf.reshape(act[0], [self.n_candidates, -1, self.action_space_dims])  # (n_candidates, batch_size, act_dims)
        cand_a = tf.transpose(cand_a, perm=[1, 0, 2])  # (batch_size, n_candidates, act_dims)
        self.optimal_action = tf.squeeze(tf.batch_gather(cand_a, idx), axis=1)
        """
        returns = tf.reshape(returns, (-1, self.n_candidates))  # (batch_size, n_candidates)
        cand_a = tf.reshape(act[0], [-1, self.n_candidates, self.action_space_dims])  # (batch_size, n_candidates, act_dims)
        idx = tf.reshape(tf.argmax(returns, axis=1), [-1, 1])  # (batch_size, 1)
        self.optimal_action = tf.squeeze(tf.batch_gather(cand_a, idx), axis=1)

    def get_rs_action(self, observations):
        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        returns = np.zeros((n * m,))

        a = self.get_random_action(h * n * m).reshape((h, n * m, -1))

        cand_a = a[0].reshape((m, n, -1))
        observation = np.repeat(observations, n, axis=0)
        for t in range(h):
            next_observation = self.dynamics_model.predict(observation, a[t])
            if self.use_reward_model:
                assert self.reward_model is not None
                rewards = self.reward_model.predict(observation, a[t], next_observation)
            else:
                rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation
        returns = returns.reshape(m, n)
        return cand_a[range(m), np.argmax(returns, axis=1)]

    def _get_rs_action(self, observations):
        sess = tf.get_default_session()
        return sess.run(self.optimal_action, feed_dict={self.obs_ph: observations})

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        pass

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
