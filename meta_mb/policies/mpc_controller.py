from meta_mb.utils.serializable import Serializable
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.optimizers.mpc_tau_optimizer import MPCTauOptimizer
import tensorflow as tf
import numpy as np


class MPCController(Serializable):
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
        self.initializer_str = initializer_str
        self.reg_coef = reg_coef
        self.reg_str = reg_str
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.num_cem_iters = num_cem_iters
        self.num_opt_iters = num_opt_iters
        self.opt_learning_rate = opt_learning_rate
        self.num_envs = num_rollouts
        self.percent_elites = percent_elites
        self.num_elites = int(percent_elites * n_candidates)
        self.env = env
        self.use_reward_model = use_reward_model
        self.alpha = alpha
        self.num_particles = num_particles
        self.clip_norm = clip_norm

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1
        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_envs, self.obs_space_dims), name='obs')
        if method_str == 'opt_policy':
            self.policy = GaussianMLPPolicy(
                name='gaussian-mlp-policy',
                obs_dim=self.obs_space_dims,
                action_dim=self.action_space_dims,
                hidden_sizes=(64, 64),
                learn_std=True,
                hidden_nonlinearity=tf.tanh,  # TODO: tunable?
                output_nonlinearity=None,  # TODO: scale to match action space range later
            )
            self.tau_optimizer = MPCTauOptimizer(
                max_epochs=num_opt_iters,
                learning_rate=opt_learning_rate,
                clip_norm=clip_norm,
            )
            self.build_opt_graph_w_policy()
        elif method_str == 'opt_act':
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
            raise NotImplementedError('method_str must be in [opt_policy, opt_act, cem, rs]')

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]
        return self.get_actions(observation)

    def get_actions(self, observations, return_first_info=False, log_grads_for_plot=False):
        agent_infos = []

        if self.method_str == 'opt_policy':
            result = self.tau_optimizer.optimize(
                {'obs': observations},
                run_extra_result_op=return_first_info,
                log_grads_for_plot=log_grads_for_plot,
            )
            if return_first_info:
                actions, neg_returns, reg, mean, std = result
                agent_infos = [dict(mean=mean, std=std, reg=reg)]
            else:
                actions, neg_returns, reg = result

        elif self.method_str == 'opt_act':
            # info to plot action executed in the fist env (observation)
            result = self.tau_optimizer.optimize(
                {'obs': observations, 'tau_mean': self.tau_mean_val},
                run_extra_result_op=return_first_info,
                log_grads_for_plot=log_grads_for_plot,
            )
            if return_first_info:
                actions, tau_mean_val, neg_returns, reg, mean, std = result
                agent_infos = [dict(mean=mean, std=std, reg=reg)]
            else:
                actions, tau_mean_val, neg_returns, reg = result

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

        else:  # CEM or RS
            sess = tf.get_default_session()
            actions, = sess.run([self.optimal_action], feed_dict={self.obs_ph: observations})
            if return_first_info:
                agent_infos = [dict(mean=actions[0], std=np.zeros_like(actions[0]), reg=0)]

        return actions, agent_infos

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

    def build_rs_graph(self):
        # FIXME: not sure if it workers for batch_size > 1 (num_rollouts > 1)
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
            next_observation = self.dynamics_model.predict_sym(observation, act[t])
            assert self.reward_model is None
            rewards = self.unwrapped_env.tf_reward(observation, act[t], next_observation)
            returns += self.discount ** t * rewards
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

    def build_cem_graph(self):
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
            observation = tf.reshape(
                tf.tile(tf.expand_dims(self.obs_ph, -1), [1, self.n_candidates, 1]),
                [-1, self.obs_space_dims]
            )
            act = tf.reshape(act, shape=[self.horizon, tf.shape(self.obs_ph)[0] * self.n_candidates,
                                         self.action_space_dims])
            for t in range(self.horizon):
                next_observation = self.dynamics_model.predict_sym(observation, act[t])
                assert self.reward_model is None
                rewards = self.unwrapped_env.tf_reward(observation, act[t], next_observation)
                returns += self.discount ** t * rewards
                observation = next_observation

            # Re-fit belief to the best ones.
            returns = tf.reshape(returns, (tf.shape(self.obs_ph)[0], self.n_candidates))  # (batch_size, n_candidates)
            act = tf.reshape(act, shape=[self.horizon, tf.shape(self.obs_ph)[0], self.n_candidates,
                                         self.action_space_dims])
            _, indices = tf.nn.top_k(returns, self.num_elites, sorted=False)
            act = tf.transpose(act, (1, 2, 3, 0))  # (batch_size, n_candidates, obs_dim, horizon)
            elite_actions = tf.batch_gather(act, indices)
            elite_actions = tf.transpose(elite_actions, (3, 0, 1, 2))  # (horizon, batch_size, n_candidates, obs_dim)
            elite_mean, elite_var = tf.nn.moments(elite_actions, axes=[2])
            elite_mean, elite_var = tf.expand_dims(elite_mean, axis=2), tf.expand_dims(elite_var, axis=2)
            mean = mean * self.alpha + (1 - self.alpha) * elite_mean
            var = var * self.alpha + (1 - self.alpha) * elite_var

        self.optimal_action = tf.squeeze(mean[0], axis=1)

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
        tau = tf.tanh(tau)  # FIXME: scale to (self.env.action_space.low, high)
        # TODO: rather than clipping, add penalty to actions outside valid range
        obs = self.obs_ph
        # returns, reg = tf.zeros(shape=(self.num_envs,)), tf.zeros(shape=(self.num_envs,))
        returns, reg = 0, 0

        if not self.reg_str and self.dyn_pred_str == 'rand':   # transition ~ f^(i) where i = 1...T, f^(i) randomly picked
            for t in range(self.horizon):
                acts = tau[t]
                next_obs = self.dynamics_model.predict_sym(obs, acts)
                rewards = self.unwrapped_env.tf_reward(obs, acts, next_obs)
                returns += self.discount ** t * tf.reduce_mean(rewards, axis=0)
                obs = next_obs
        else:  # compute transition ~ f_j for all f_j in the model ensemble
            if self.dyn_pred_str == 'all':
                obs = [obs for _ in range(self.dynamics_model.num_models)]
                obs = tf.concat(obs, axis=0)
            for t in range(self.horizon):
                acts = tau[t]
                if self.dyn_pred_str == 'all':
                    acts = [acts for _ in range(self.dynamics_model.num_models)]
                    acts = tf.concat(acts, axis=0)
                # if self.reg_str == 'uncertainty':
                #     pred_obs = self.dynamics_model.predict_sym_all(
                #         obs, acts, same_obs=(t==0), same_act=True, reg_str=self.reg_str, pred_type=self.dyn_pred_str,
                #     ) # (num_envs, obs_space_dims, num_models)
                #     uncertainty = tf.math.reduce_variance(pred_obs, axis=-1)
                #     uncertainty = tf.reduce_sum(uncertainty, axis=1)
                #     reg += uncertainty
                #     # batch_gather params = (num_envs, num_models, obs_space_dims), indices = (num_envs, 1)
                #     idx = tf.random.uniform(shape=(self.num_envs,), minval=0, maxval=self.dynamics_model.num_models, dtype=tf.int32)
                #     next_obs = tf.batch_gather(tf.transpose(pred_obs, (0, 2, 1)), tf.reshape(idx, [-1, 1]))
                #     next_obs = tf.squeeze(next_obs, axis=1)
                # else:
                #     if self.dyn_pred_str == 'batches':
                #         next_obs = self.dynamics_model.predict_batches_sym(obs, acts)
                #     elif self.dyn_pred_str == 'rand':
                #         next_obs = self.dynamics_model.predict_sym(obs, acts)
                #     else:
                #         next_obs = self.dynamics_model.predict_sym_all(obs, acts, pred_type=self.dyn_pred_str)

                next_obs, one_step_reg = self.dynamics_model.predict_sym_all(
                    obs, acts, reg_str=self.reg_str, pred_type=self.dyn_pred_str,
                )
                reg += tf.reduce_mean(one_step_reg, axis=0)

                rewards = self.unwrapped_env.tf_reward(obs, acts, next_obs)
                returns += self.discount ** t * tf.reduce_mean(rewards, axis=0)
                obs = next_obs

        # build loss = total_cost + regularization
        # neg_returns = tf.reduce_mean(-returns, axis=0)
        # reg = tf.reduce_mean(reg, axis=0)
        neg_returns = -returns

        result_op = [tau[0], mean_var, neg_returns, self.reg_coef*reg]
        extra_result_op = [mean_var[0][0], tf.exp(log_std_var[0][0])]

        self.tau_optimizer.build_graph(
            loss=neg_returns+self.reg_coef*reg,
            init_op=[tf.assign(mean_var, self.tau_mean_ph)],
            var_list=[mean_var, log_std_var],
            result_op=result_op,
            extra_result_op=extra_result_op,
            input_ph_dict={'obs': self.obs_ph, 'tau_mean': self.tau_mean_ph},
        )

    def predict_open_loop(self, init_obs, tau):
        obs_hall, obs_hall_mean, obs_hall_std, reward_hall = [], [], [], []
        obs = init_obs
        for action in tau:
            next_obs, agent_info = self.dynamics_model.predict(
                obs[None],
                action[None],
                pred_type='rand',
                deterministic=False,
                return_infos=True,
            )
            next_obs, agent_info = next_obs[0], agent_info[0]
            obs_hall.append(next_obs)
            obs_hall_mean.append(agent_info['mean'])
            obs_hall_std.append(agent_info['std'])
            reward_hall.extend(self.env.reward(obs[None], action[None], next_obs[None]))
            obs = next_obs
        return obs_hall, obs_hall_mean, obs_hall_std, reward_hall

    def build_opt_graph_w_policy(self):
        # returns, reg = tf.zeros(shape=(self.num_envs,)), tf.zeros(shape=(self.num_envs,))
        returns, reg = 0, 0
        obs = self.obs_ph


        if not self.reg_str and self.dyn_pred_str == 'rand':   # transition ~ f^(i) where i = 1...T, f^(i) randomly picked
            for t in range(self.horizon):
                dist_policy = self.policy.distribution_info_sym(obs)
                acts, dist_policy = self.policy.distribution.sample_sym(dist_policy)
                acts = tf.tanh(acts)
                if t == 0:
                    result_op = [acts]
                    extra_result_op = [dist_policy['mean'][0], tf.exp(dist_policy['log_std'][0])]

                next_obs = self.dynamics_model.predict_sym(obs, acts)
                rewards = self.unwrapped_env.tf_reward(obs, acts, next_obs)
                returns += self.discount ** t * tf.reduce_mean(rewards, axis=0)
                obs = next_obs
        else:  # compute transition ~ f_j for all f_j in the model ensemble
            if self.dyn_pred_str == 'all':
                obs = [obs for _ in range(self.dynamics_model.num_models)]
                obs = tf.concat(obs, axis=0)
            for t in range(self.horizon):
                dist_policy = self.policy.distribution_info_sym(obs)
                acts, dist_policy = self.policy.distribution.sample_sym(dist_policy)
                acts = tf.tanh(acts)
                if t == 0:
                    result_op = [acts]
                    extra_result_op = [dist_policy['mean'][0], tf.exp(dist_policy['log_std'][0])]

                if self.dyn_pred_str == 'all':
                    acts = [acts for _ in range(self.dynamics_model.num_models)]
                    acts = tf.concat(acts, axis=0)

                next_obs, one_step_reg = self.dynamics_model.predict_sym_all(
                    obs, acts, reg_str=self.reg_str, pred_type=self.dyn_pred_str,
                )
                reg += tf.reduce_mean(one_step_reg, axis=0)

                rewards = self.unwrapped_env.tf_reward(obs, acts, next_obs)
                returns += self.discount ** t * tf.reduce_mean(rewards, axis=0)
                obs = next_obs

        # build loss = total_cst + regularization
        # neg_returns = tf.reduce_mean(-returns, axis=0)
        # if self.reg_str == 'uncertainty' or self.reg_str is None:
        #     reg = tf.reduce_mean(reg, axis=0)
        # else:
        #     raise NotImplementedError
        neg_returns = -returns

        result_op += [neg_returns, self.reg_coef*reg]

        self.tau_optimizer.build_graph(
            loss=neg_returns+self.reg_coef*reg,
            var_list=list(self.policy.get_params().values()),
            result_op=result_op,
            extra_result_op=extra_result_op,
            input_ph_dict={'obs': self.obs_ph},
            with_policy=True,
            #lmbda=lmbda,
            #loss_dual=tf.reduce_mean(-lmbda_kl, axis=0),
        )

    def get_cem_action(self, observations):
        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        act_dim = self.env.action_space.shape[0]

        num_elites = max(int(self.n_candidates * self.percent_elites), 1)
        mean = np.ones((m, h * act_dim)) * (self.env.action_space.high + self.env.action_space.low) / 2
        std = np.ones((m, h * act_dim)) * (self.env.action_space.high - self.env.action_space.low) / 16
        clip_low = np.concatenate([self.env.action_space.low] * h)
        clip_high = np.concatenate([self.env.action_space.high] * h)

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
            lb_dist, ub_dist = mean - self.env.action_space.low, self.env.action_space.high - mean
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
            if self.use_reward_model:
                assert self.reward_model is not None
                rewards = self.reward_model.predict(observation, a[t], next_observation)
            else:
                rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation
        returns = returns.reshape(m, n)
        return cand_a[range(m), np.argmax(returns, axis=1)]

    def plot_grads(self):
        if self.method_str in ['opt_act', 'opt_policy']:
            self.tau_optimizer.plot_grads()

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        if self.method_str == 'opt_act':
            assert len(dones) == self.num_envs
            if self.initializer_str == 'uniform':
                self.tau_mean_val = np.random.uniform(
                    low=self.env.action_space.low,
                    high=self.env.action_space.high,
                    size=(self.horizon, self.num_envs, self.action_space_dims),
                )
            elif self.initializer_str == 'zeros':
                self.tau_mean_val = np.zeros(
                    (self.horizon, self.num_envs, self.action_space_dims),
                )

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
