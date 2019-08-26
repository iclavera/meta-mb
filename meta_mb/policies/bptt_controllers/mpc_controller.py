from meta_mb.utils.serializable import Serializable
import tensorflow as tf
import numpy as np


class MPCController(Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            method_str='cem',
            num_rollouts=None,
            discount=1,
            n_candidates=1024,
            horizon=10,
            num_cem_iters=8,
            percent_elites=0.1,
            alpha=0.1,
            num_particles=20,
    ):
        Serializable.quick_init(self, locals())
        self.dynamics_model = dynamics_model
        self.discount = discount
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.num_cem_iters = num_cem_iters
        self.num_envs = num_rollouts
        self.percent_elites = percent_elites
        self.num_elites = int(percent_elites * n_candidates)
        self.env = env
        self.alpha = alpha
        self.num_particles = num_particles

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_envs, self.obs_dim), name='obs')
        self.optimized_actions = None
        if method_str == 'cem':
            self.build_cem_graph()
        elif method_str == 'rs':
            self.build_rs_graph()
        else:
            raise NotImplementedError('method_str must be in [opt_policy, opt_act, cem, rs]')

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        assert observation.ndim == 1
        return self.get_actions(observation[None])

    def get_actions(self, observations, verbose=True):
        agent_infos = []

        sess = tf.get_default_session()
        actions, = sess.run([self.optimized_actions], feed_dict={self.obs_ph: observations})

        return actions, agent_infos

    def get_random_action(self, n):
        return np.random.uniform(low=self.act_low,
                                 high=self.act_high, size=(n,) + self.act_low.shape)

    def get_sinusoid_actions(self, action_space, t):
        actions = np.array([])
        for i in range(action_space):
            actions = np.append(actions, 0.5 * np.sin(i * t))
        return actions

    def build_rs_graph(self):
        n_candidates = self.n_candidates
        num_envs = self.num_envs
        horizon = self.horizon

        returns = tf.zeros((num_envs*n_candidates,))
        # (num_envs, obs_dim) => (num_envs, 1, obs_dim) => (num_envs, n_candidates, obs_dim) => (num_envs*n_candidates, obs_dim)
        obs = tf.reshape(
            tf.tile(tf.expand_dims(self.obs_ph, 1), [1, n_candidates, 1]),
            shape=(num_envs*n_candidates, self.obs_dim)
        )
        act = tf.random.uniform(
            shape=(horizon, num_envs*n_candidates, self.act_dim),
            minval=self.act_low, maxval=self.act_high,
        )

        for t in range(horizon):
            next_obs = self.dynamics_model.predict_sym(obs, act[t])
            rewards = self.unwrapped_env.tf_reward(obs, act[t], next_obs)
            returns += self.discount**t * rewards
            obs = next_obs

        returns = tf.reshape(returns, (num_envs, n_candidates))
        _, indices = tf.nn.top_k(returns, k=1, sorted=False)
        cand_a = tf.reshape(act[0], shape=(num_envs, n_candidates, self.act_dim))
        self.optimized_actions = tf.squeeze(tf.batch_gather(cand_a, indices), axis=1)

    def build_cem_graph(self):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        horizon = self.horizon
        n_candidates = self.n_candidates
        num_envs = self.num_envs

        mean = tf.ones(shape=[horizon, num_envs, 1, act_dim]) * (self.act_high + self.act_low) / 2
        var = tf.ones(shape=[horizon, num_envs, 1, act_dim]) * (self.act_high - self.act_low) / 16

        for itr in range(self.num_cem_iters):
            lb_dist, ub_dist = mean - self.act_low, self.act_high - mean
            constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
            std = tf.sqrt(constrained_var)
            act = mean + tf.random.normal(shape=[horizon, num_envs, n_candidates, act_dim]) * std
            act = tf.clip_by_value(act, self.act_low, self.act_high)
            returns = 0

            # (num_envs, obs_dim) => (num_envs, obs_dim, 1) => (num_envs, obs
            obs = tf.reshape(
                tf.tile(tf.expand_dims(self.obs_ph, axis=1), [1, n_candidates, 1]),
                shape=(num_envs*n_candidates, obs_dim),
            )
            act = tf.reshape(act, shape=(horizon, num_envs*n_candidates, act_dim))
            for t in range(horizon):
                next_obs = self.dynamics_model.predict_sym(obs, act[t])
                rewards = self.unwrapped_env.tf_reward(obs, act[t], next_obs)
                returns += self.discount**t * rewards
                obs = next_obs

            # Re-fit belief to the best ones
            returns = tf.reshape(returns, (num_envs, n_candidates))
            _, indices = tf.nn.top_k(returns, k=self.num_elites, sorted=False)
            act = tf.reshape(act, shape=(horizon, num_envs, n_candidates, act_dim))
            act = tf.transpose(act, (1, 2, 3, 0))  # (num_envs, n_candidates, act_dim, horizon)
            elite_actions = tf.batch_gather(act, indices)
            elite_actions = tf.transpose(elite_actions, (3, 0, 1, 2))  # (horizon, num_envs, n_candidates, act_dim)
            elite_mean, elite_var = tf.nn.moments(elite_actions, axes=[2], keep_dims=True)
            mean = mean * self.alpha + elite_mean * (1 - self.alpha)
            var = var * self.alpha + elite_var * (1 - self.alpha)

        self.optimized_actions = mean[0, :, 0, :]

    def get_cem_action(self, observations):
        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        act_dim = self.env.action_space.shape[0]

        num_elites = max(int(self.n_candidates * self.percent_elites), 1)
        mean = np.ones((m, h * act_dim)) * (self.act_high + self.act_low) / 2
        std = np.ones((m, h * act_dim)) * (self.act_high - self.act_low) / 16
        clip_low = np.concatenate([self.act_low] * h)
        clip_high = np.concatenate([self.act_high] * h)

        for i in range(self.num_cem_iters):
            z = np.random.normal(size=(n, m, h * act_dim))
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
                                       self.num_particles, axis=-1), axis=0)  # TODO: Make sure this reshaping works
            elites_idx = ((-returns).argsort(axis=-1) < num_elites).T
            elites = a_stacked[elites_idx]
            mean = mean * self.alpha + (1 - self.alpha) * np.mean(elites, axis=0)
            std = np.std(elites, axis=0)
            lb_dist, ub_dist = mean - self.act_low, self.act_high - mean
            std = np.minimum(np.minimum(lb_dist / 2, ub_dist / 2), std)

        return cand_a[range(m), np.argmax(returns, axis=1)]

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
            rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation
        returns = returns.reshape(m, n)
        return cand_a[range(m), np.argmax(returns, axis=1)]

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
