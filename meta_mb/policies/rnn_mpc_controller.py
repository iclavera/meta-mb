from meta_mb.utils.serializable import Serializable
import numpy as np
import tensorflow as tf


class RNNMPCController(Serializable):
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
            percent_elites=0.05,
            use_reward_model=False,
            alpha=0.1,
            use_graph=True,
    ):
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
        self._hidden_state = None
        self.use_graph = use_graph

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, 'wrapped_env'):
            self.unwrapped_env = self.unwrapped_env.wrapped_env

        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]

        # make sure that env has reward function
        if not self.use_reward_model:
            assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        Serializable.quick_init(self, locals())

        if use_graph:
            self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_space_dims), name='obs')
            self.hidden_state_c_ph = tf.placeholder(dtype=tf.float32, shape=(None, ) + dynamics_model.hidden_sizes)
            self.hidden_state_h_ph = tf.placeholder(dtype=tf.float32, shape=(None, ) + dynamics_model.hidden_sizes)
            self.hidden_state_ph = tf.nn.rnn_cell.LSTMStateTuple(self.hidden_state_c_ph, self.hidden_state_h_ph)
            self.optimal_action = None
            if not use_cem:
                self.build_rs_graph()
            else:
                self.build_cem_graph()

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]

        action = self.get_actions(observation)[0]

        return action, dict()

    def get_actions(self, observations):
        if self.use_graph:
            sess = tf.get_default_session()
            actions = sess.run(self.optimal_action, feed_dict={self.obs_ph: observations,
                                                               self.hidden_state_c_ph: self._hidden_state.c,
                                                               self.hidden_state_h_ph: self._hidden_state.h})
        else:
            if self.use_cem:
                actions = self.get_cem_action(observations)
            else:
                actions = self.get_rs_action(observations)

        _, self._hidden_state = self.dynamics_model.predict(np.array(observations), actions, self._hidden_state)

        return actions, dict()

    def get_random_action(self, n):
        return np.random.uniform(low=self.env.action_space.low,
                                 high=self.env.action_space.high, size=(n,) + self.env.action_space.low.shape)

    def get_cem_action(self, observations):

        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        act_dim = self.env.action_space.shape[0]

        num_elites = max(int(self.n_candidates * self.percent_elites), 1)
        mean = np.zeros((m, h * act_dim))
        std = np.ones((m, h * act_dim))
        clip_low = np.concatenate([self.env.action_space.low] * h)
        clip_high = np.concatenate([self.env.action_space.high] * h)

        for i in range(self.num_cem_iters):
            z = np.random.normal(size=(n, m,  h * act_dim))
            a = mean + z * std
            a_stacked = np.clip(a, clip_low, clip_high)
            a = a.reshape((n * m, h, act_dim))
            a = np.transpose(a, (1, 0, 2))
            returns = np.zeros((n * m,))

            cand_a = a[0].reshape((m, n, -1))
            observation = np.repeat(observations, n, axis=0)
            hidden_state = self.repeat_hidden(self._hidden_state, n)
            for t in range(h):
                next_observation, hidden_state = self.dynamics_model.predict(observation, a[t], hidden_state)
                if self.use_reward_model:
                    assert self.reward_model is not None
                    rewards = self.reward_model.predict(observation, a[t], next_observation)
                else:
                    rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
                returns += self.discount ** t * rewards
                observation = next_observation
            returns = returns.reshape(m, n)
            elites_idx = ((-returns).argsort(axis=-1) < num_elites).T
            elites = a_stacked[elites_idx]
            mean = np.mean(elites, axis=0)
            std = np.std(elites, axis=0)

        return cand_a[range(m), np.argmax(returns, axis=1)]

    def build_cem_graph(self):
        num_elites = max(int(self.n_candidates * self.percent_elites), 1)

        mean = tf.ones(shape=[self.horizon, tf.shape(self.obs_ph)[0], 1,
                              self.action_space_dims]) * (self.env.action_space.high + self.env.action_space.low) / 2
        var = tf.ones(shape=[self.horizon, tf.shape(self.obs_ph)[0], 1,
                             self.action_space_dims]) * (self.env.action_space.high - self.env.action_space.low) / 16

        for itr in range(self.num_cem_iters):
            lb_dist, ub_dist = mean - self.env.action_space.low, self.env.action_space.high - mean
            constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
            std = tf.sqrt(constrained_var)
            act = mean + tf.random.normal(shape=[self.horizon, tf.shape(self.obs_ph)[0], self.n_candidates,
                                                 self.action_space_dims]) * std
            act = tf.clip_by_value(act, self.env.action_space.low, self.env.action_space.high)
            returns = 0
            observation = self.repeat_sym(self.obs_ph, self.n_candidates)
            hidden_state = self.repeat_hidden_sym(self.hidden_state_ph, self.n_candidates)

            act = tf.reshape(act, shape=[self.horizon, tf.shape(self.obs_ph)[0] * self.n_candidates,
                                         self.action_space_dims])
            for t in range(self.horizon):
                next_observation, hidden_state = \
                    self.dynamics_model.predict_sym(tf.expand_dims(observation, -2),
                                                    tf.expand_dims(act[t], -2),
                                                    hidden_state)
                if not self.use_reward_model:
                    assert self.reward_model is None
                    rewards = self.unwrapped_env.tf_reward(observation, act[t], next_observation)
                else:
                    assert not (self.reward_model is None)
                    rewards = self.reward_model.predict_sym(observation, act[t], next_observation)
                returns += self.discount ** t * rewards
                observation = next_observation

            # Re-fit belief to the best ones.
            returns = tf.reshape(returns, (tf.shape(self.obs_ph)[0], self.n_candidates))  # (batch_size, n_candidates)
            act = tf.reshape(act, shape=[self.horizon, tf.shape(self.obs_ph)[0], self.n_candidates,
                                         self.action_space_dims])
            _, indices = tf.nn.top_k(returns, num_elites, sorted=False)
            act = tf.transpose(act, (1, 2, 3, 0))  # (batch_size, n_candidates, obs_dim, horizon)
            elite_actions = tf.batch_gather(act, indices)
            elite_actions = tf.transpose(elite_actions, (3, 0, 1, 2))  # (horizon, batch_size, n_candidates, obs_dim)
            elite_mean, elite_var = tf.nn.moments(elite_actions, axes=[2])
            elite_mean, elite_var = tf.expand_dims(elite_mean, axis=2), tf.expand_dims(elite_var, axis=2)
            # mean = mean * self.alpha + (1 - self.alpha) * elite_mean
            # var = var * self.alpha + (1 - self.alpha) * elite_var
            mean, var = elite_mean, elite_var


        self.optimal_action = tf.squeeze(mean[0], axis=1)

    def get_rs_action(self, observations):
        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        returns = np.zeros((n * m,))

        a = self.get_random_action(h * n * m).reshape((h, n * m, -1))

        cand_a = a[0].reshape((m, n, -1))
        observation = np.repeat(observations, n, axis=0)
        hidden_state = self.repeat_hidden(self._hidden_state, n)

        for t in range(h):
            next_observation, hidden_state = self.dynamics_model.predict(observation, a[t], hidden_state)
            if self.use_reward_model:
                assert self.reward_model is not None
                rewards = self.reward_model.predict(observation, a[t], next_observation)
            else:
                rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation
        returns = returns.reshape(m, n)
        return cand_a[range(m), np.argmax(returns, axis=1)]

    def build_rs_graph(self):
        # FIXME: not sure if it workers for batch_size > 1 (num_rollouts > 1)
        returns = 0  # (batch_size * n_candidates,)
        act = tf.random.uniform(
            shape=[self.horizon, tf.shape(self.obs_ph)[0] * self.n_candidates, self.action_space_dims],
            minval=self.env.action_space.low,
            maxval=self.env.action_space.high)

        # Equivalent to np.repeat
        observation = self.repeat_sym(self.obs_ph, self.n_candidates)

        hidden_state = self.repeat_hidden_sym(self.hidden_state_ph, self.n_candidates)


        for t in range(self.horizon):
            # dynamics_dist = self.dynamics_model.distribution_info_sym(observation, act[t])
            # mean, var = dynamics_dist['mean'], dynamics_dist['var']
            # next_observation = mean + tf.random.normal(shape=tf.shape(mean))*tf.sqrt(var)
            next_observation, hidden_state = \
                self.dynamics_model.predict_sym(tf.expand_dims(observation, -2),
                                                tf.expand_dims(act[t], -2),
                                                hidden_state)

            if not self.use_reward_model:
                assert self.reward_model is None
                rewards = self.unwrapped_env.tf_reward(observation, act[t], next_observation)
            else:
                assert not (self.reward_model is None)
                rewards = self.reward_model.predict_sym(observation, act[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation

        returns = tf.reshape(returns, (-1, self.n_candidates))  # (batch_size, n_candidates)
        cand_a = tf.reshape(act[0], [-1, self.n_candidates, self.action_space_dims])  # (batch_size, n_candidates, act_dims)
        idx = tf.reshape(tf.argmax(returns, axis=1), [-1, 1])  # (batch_size, 1)
        self.optimal_action = tf.squeeze(tf.batch_gather(cand_a, idx), axis=1)

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple

        if dones is None:
            dones = [True]
        if self._hidden_state is None:
            self._hidden_state = self.dynamics_model.get_initial_hidden(batch_size=len(dones), batch=False)

        zero_hidden_state = self.dynamics_model.get_initial_hidden(batch_size=1, batch=False)
        if isinstance(zero_hidden_state, LSTMStateTuple):
            self._hidden_state.c[dones] = zero_hidden_state.c
            self._hidden_state.h[dones] = zero_hidden_state.h
        elif len(zero_hidden_state) == 1:
            self._hidden_state[dones] = zero_hidden_state
        else:
            for i, z_h in enumerate(zero_hidden_state):
                if isinstance(z_h, LSTMStateTuple):
                    self._hidden_state[i].c[dones] = z_h.c
                    self._hidden_state[i].h[dones] = z_h.h
                else:
                    self._hidden_state[i][dones] = z_h

    def repeat_hidden(self, hidden, n):
        LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
        if not isinstance(hidden, list) and not isinstance(hidden, tuple):
            if isinstance(hidden, LSTMStateTuple):
                hidden_c = hidden.c
                hidden_h = hidden.h
                return LSTMStateTuple(np.repeat(hidden_c, n, axis=0), np.repeat(hidden_h, n, axis=0))

            else:
                return np.repeat(hidden, n, axis=0)
        else:
            _hidden = []
            for h in hidden:
                _h = self.repeat_hidden(h, n)
                _hidden.append(_h)
                # if isinstance(h, LSTMStateTuple):
                #     hidden_c = h.c
                #     hidden_h = h.h
                #     _hidden.append(LSTMStateTuple(np.repeat(hidden_c, n, axis=0), np.repeat(hidden_h, n, axis=0)))
                #
                # else:
                #     _hidden.append(np.repeat(h, n, axis=0))
            return _hidden

    def repeat_hidden_sym(self, hidden, n):
        LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
        # if not isinstance(hidden, list) and not isinstance(hidden, tuple):
        if isinstance(hidden, LSTMStateTuple):
            hidden_c = hidden.c
            hidden_h = hidden.h
            return LSTMStateTuple(self.repeat_sym(hidden_c, n), self.repeat_sym(hidden_h, n))

        #     else:
        #         return self.repeat_sym(hidden, n)
        # else:
        #     _hidden = []
        #     for h in hidden:
        #         _h = self.repeat_hidden(h, n)
        #         _hidden.append(_h)
        #     return _hidden

    def repeat_sym(self, tensor, n_times):
        """
        :param tensor: two dimensional tensor nxd
        :return: a (n x n_times x d) tensor
        """
        ret = tf.reshape(
            tf.tile(tf.expand_dims(tensor, -1), [1, n_times, 1]),
            [-1, tensor.shape[-1]]
        )

        return ret

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        # state = LayersPowered.__getstate__(self)
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        # LayersPowered.__setstate__(self, state)
        Serializable.__setstate__(self, state['init_args'])
