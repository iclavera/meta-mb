from meta_mb.utils.serializable import Serializable
from meta_mb.optimizers.mpc_tau_optimizer import MPCTauOptimizer
import numpy as np
import tensorflow as tf


class RNNMPCController(Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            num_rollouts=None,
            reward_model=None,
            discount=1,
            method_str='opt_policy',
            dyn_pred_str='rand',
            policy_str='gaussian_mlp',
            initializer_str='uniform',
            reg_coef=1,
            reg_str=None,
            n_candidates=1024,
            horizon=10,
            num_cem_iters=8,
            num_opt_iters=8,
            opt_learning_rate=1e-3,
            clip_norm=-1,
            percent_elites=0.1,
            use_reward_model=False,
            alpha=0.1,
            num_particles=20,
    ):
        Serializable.quick_init(self, locals())
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.method_str = method_str
        self.dyn_pred_str = dyn_pred_str
        self.num_envs = num_rollouts
        self.policy_str = policy_str
        self.initializer_str = initializer_str
        self.reg_coef = reg_coef
        self.reg_str = reg_str
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.num_cem_iters = num_cem_iters
        self.percent_elites = percent_elites
        self.num_elites = int(percent_elites * n_candidates)
        self.env = env
        self.use_reward_model = use_reward_model
        self.alpha = alpha
        self.num_particles = num_particles

        self._hidden_state = None

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, 'wrapped_env'):
            self.unwrapped_env = self.unwrapped_env.wrapped_env

        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]

        # make sure that env has reward function
        if not self.use_reward_model:
            assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_envs, self.obs_space_dims), name='obs')
        if method_str != 'opt_act' or (not self.reg_str and self.dyn_pred_str == 'rand'):  # use predict_sym
            self.hidden_state_c_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_envs,) + dynamics_model.hidden_sizes)
            self.hidden_state_h_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_envs,) + dynamics_model.hidden_sizes)
        else:  # use predict_sym_all
            self.hidden_state_c_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_envs*self.dynamics_model.num_models,) + dynamics_model.hidden_sizes)
            self.hidden_state_h_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_envs*self.dynamics_model.num_models,) + dynamics_model.hidden_sizes)
        self.hidden_state_ph = tf.nn.rnn_cell.LSTMStateTuple(self.hidden_state_c_ph, self.hidden_state_h_ph)

        if method_str == 'opt_act':
            if initializer_str == 'uniform':
                self.tau_mean_val = np.random.uniform(
                    low=self.env.action_space.low,
                    high=self.env.action_space.high,
                    size=(self.horizon, self.num_envs, self.action_space_dims),
                )
            elif initializer_str == 'zeros':
                self.tau_mean_val = np.zeros(
                    (self.horizon, self.num_envs, self.action_space_dims),
                )
            else:
                raise NotImplementedError('initializer_str must be uniform or zeros')
            self.tau_mean_ph = tf.placeholder(
                dtype=tf.float32,
                shape=np.shape(self.tau_mean_val),
                name='tau_mean',
            )
            self.tau_optimizer = MPCTauOptimizer(
                max_epochs=num_opt_iters,
                learning_rate=opt_learning_rate,
                clip_norm=clip_norm,
            )
            self.build_opt_graph()
        elif method_str == 'cem':
            self.build_cem_graph()
        elif method_str == 'rs':
            self.build_rs_graph()
        else:
            raise NotImplementedError('method_str must be in [opt_act, cem, rs]')

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]
        return self.get_actions(observation)

    def get_actions(self, observations, return_first_info=False, log_grads_for_plot=False):
        agent_infos = []

        if self.method_str == 'opt_act':
            # info to plot action executed in the fist env (observation)
            result = self.tau_optimizer.optimize(
                {'obs': observations, 'tau_mean': self.tau_mean_val, 'hidden_c': self._hidden_state.c, 'hidden_h': self._hidden_state.h},
                run_extra_result_op=return_first_info,
                log_grads_for_plot=log_grads_for_plot,
            )
            if return_first_info:
                actions, next_hidden, tau_mean_val, neg_returns, reg, mean, std = result
                agent_infos = [dict(mean=mean, std=std, reg=reg)]
            else:
                actions, next_hidden, tau_mean_val, neg_returns, reg = result

            # rotate
            if self.initializer_str == 'uniform':
                self.tau_mean_val = np.concatenate([
                    tau_mean_val[1:],
                    np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high, size=(1, self.num_envs, self.action_space_dims)),
                ], axis=0)
            else:
                self.tau_mean_val = np.concatenate([
                    tau_mean_val[1:],
                    np.zeros((1, self.num_envs, self.action_space_dims)),
                ], axis=0)
        else:
            sess = tf.get_default_session()
            actions, next_hidden = sess.run([self.optimal_action, self.next_hidden],
                                            feed_dict={self.obs_ph: observations,
                                                       self.hidden_state_c_ph: self._hidden_state.c,
                                                       self.hidden_state_h_ph: self._hidden_state.h})
            if return_first_info:
                agent_infos = [dict(mean=actions[0], std=np.zeros_like(actions[0]), reg=0)]

        # _, self._hidden_state = self.dynamics_model.predict(np.array(observations), actions, self._hidden_state)
        self._hidden_state = next_hidden
        return actions, agent_infos

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
        _, self.next_hidden = \
            self.dynamics_model.predict_sym(tf.expand_dims(self.obs_ph, axis=1),
                                            tf.expand_dims(self.optimal_action, axis=1),
                                            self.hidden_state_ph)

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
            if t == 0:
                cand_c = hidden_state.c.reshape((m, n, -1))
                cand_h = hidden_state.h.reshape((m, n, -1))
            if self.use_reward_model:
                assert self.reward_model is not None
                rewards = self.reward_model.predict(observation, a[t], next_observation)
            else:
                rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation
        returns = returns.reshape(m, n)
        next_hidden = tf.nn.rnn_cell.LSTMStateTuple(cand_c[range(m), np.argmax(returns, axis=1)],
                                                    cand_h[range(m), np.argmax(returns, axis=1)])
        return cand_a[range(m), np.argmax(returns, axis=1)], next_hidden

    def build_rs_graph(self):
        returns = 0  # (batch_size * n_candidates,)
        act = tf.random.uniform(
            shape=[self.horizon, tf.shape(self.obs_ph)[0] * self.n_candidates, self.action_space_dims],
            minval=self.env.action_space.low,
            maxval=self.env.action_space.high)

        # Equivalent to np.repeat
        observation = self.repeat_sym(self.obs_ph, self.n_candidates)

        hidden_state = self.repeat_hidden_sym(self.hidden_state_ph, self.n_candidates)

        for t in range(self.horizon):
            next_observation, hidden_state = \
                self.dynamics_model.predict_sym(tf.expand_dims(observation, axis=1),
                                                tf.expand_dims(act[t], axis=1),
                                                hidden_state)
            if t == 0:
                cand_c = tf.reshape(hidden_state.c, [tf.shape(self.obs_ph)[0],
                                                     self.n_candidates,
                                                     self.dynamics_model.hidden_sizes[0]])
                cand_h = tf.reshape(hidden_state.h, [tf.shape(self.obs_ph)[0],
                                                     self.n_candidates,
                                                     self.dynamics_model.hidden_sizes[0]])

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
        self.next_hidden = tf.nn.rnn_cell.LSTMStateTuple(tf.squeeze(tf.batch_gather(cand_c, idx), axis=1),
                                                         tf.squeeze(tf.batch_gather(cand_h, idx), axis=1))

    def build_opt_graph(self):
        mean_var = tf.get_variable(
            'tau_mean',
            shape=(self.horizon, self.num_envs, self.action_space_dims),
            dtype=tf.float32,
            trainable=True,
        )

        log_std_var = tf.get_variable(
            'tau_log_std',
            shape=(1, 1, self.action_space_dims),
            dtype=tf.float32,
            initializer=tf.initializers.ones,
            trainable=True,
        )
        log_std = tf.maximum(log_std_var, np.log(5e-2))  # FIXME: hardcoded

        tau = mean_var + tf.multiply(tf.random.normal(tf.shape(mean_var)), tf.exp(log_std))
        # tau = tf.clip_by_value(tau, self.env.action_space.low, self.env.action_space.high)
        tau = tf.tanh(tau)
        obs, hidden_state = self.obs_ph, self.hidden_state_ph
        # returns, reg = tf.zeros(shape=(self.num_envs,)), tf.zeros(shape=(self.num_envs,))
        returns, reg = 0, 0

        if not self.reg_str and self.dyn_pred_str == 'rand':
            for t in range(self.horizon):
                acts = tau[t]
                next_obs = self.dynamics_model.predict_sym(obs, acts, hidden_state)
                rewards = self.unwrapped_env.tf_reward(obs, acts, next_obs)
                returns += self.discount ** t * tf.reduce_mean(rewards, axis=0)
                obs = next_obs

                if t == 0:
                    result_op = [acts, hidden_state]

        else:
            if self.dyn_pred_str == 'all':
                obs = [obs for _ in range(self.dynamics_model.num_models)]
                obs = tf.concat(obs, axis=0)
            for t in range(self.horizon):
                acts = tau[t]
                if self.dyn_pred_str == 'all':
                    acts = [acts for _ in range(self.dynamics_model.num_models)]
                    acts = tf.concat(acts, axis=0)

                next_obs, one_step_reg, hidden_state = self.dynamics_model.predict_sym_all(
                    obs, acts, hidden_state,
                    reg_str=self.reg_str, pred_type=self.dyn_pred_str,
                )
                reg += tf.reduce_mean(one_step_reg, axis=0)

                rewards = self.unwrapped_env.tf_reward(obs, acts, next_obs)
                returns += self.discount ** t * tf.reduce_mean(rewards, axis=0)
                obs = next_obs

                if t == 0:
                    result_op = [acts, hidden_state]

        # build loss = total_cost + regularization
        # neg_returns = tf.reduce_mean(-returns, axis=0)
        # reg = tf.reduce_mean(reg, axis=0)
        neg_returns = -returns

        result_op += [mean_var, neg_returns, self.reg_coef*reg]
        extra_result_op = [mean_var[0][0], tf.exp(log_std_var[0][0])]

        self.tau_optimizer.build_graph(
            loss=neg_returns+self.reg_coef*reg,
            init_op=[tf.assign(mean_var, self.tau_mean_ph)],
            var_list=[mean_var, log_std_var],
            result_op=result_op,
            extra_result_op=extra_result_op,
            input_ph_dict={'obs': self.obs_ph, 'tau_mean': self.tau_mean_ph, 'hidden_c': self.hidden_state_c_ph, 'hidden_h': self.hidden_state_h_ph},
        )

    def predict_open_loop(self, init_obs, tau):
        assert init_obs.shape == (self.obs_space_dims,)

        obs_hall, obs_hall_mean, obs_hall_std, reward_hall = [], [], [], []
        obs = init_obs
        obs_batch = np.stack([init_obs for _ in range(self.dynamics_model.num_models)])  # pretend that there is a (repeated) batch
        hidden_state_batch = self.dynamics_model.get_initial_hidden(batch_size=self.dynamics_model.num_models, batch=False)

        for action in tau:
            next_obs_batch, hidden_state_batch, agent_info = self.dynamics_model.predict(
                obs_batch,
                np.stack([action for _ in range(self.dynamics_model.num_models)]),
                hidden_state=hidden_state_batch,
            )
            next_obs = next_obs_batch[np.random.randint(low=0, high=self.dynamics_model.num_models)]
            obs_hall.append(next_obs)
            obs_hall_mean.append(np.mean(next_obs_batch, axis=0))
            obs_hall_std.append(np.std(next_obs_batch, axis=0))
            reward_hall.extend(self.env.reward(obs[None], action[None], next_obs[None]))
            obs = next_obs

        return obs_hall, obs_hall_mean, obs_hall_std, reward_hall

    def get_params_internal(self, **tags):
        return []

    def plot_grads(self):
        if self.method_str == 'opt_act':
            self.tau_optimizer.plot_grads()

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
        :return: a (n x n_times) x d tensor
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
