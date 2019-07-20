from meta_mb.utils.serializable import Serializable
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.policies.distributions.diagonal_gaussian import DiagonalGaussian
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
            use_cem=False,
            use_opt=False,
            use_opt_w_policy=False,
            initializer_str='uniform',
            kl_coef=1,
            n_candidates=1024,
            horizon=10,
            num_cem_iters=8,
            num_opt_iters=8,
            opt_learning_rate=1e-3,
            percent_elites=0.1,
            use_reward_model=False,
            alpha=0.1,
            num_particles=20,
            use_graph=True,
    ):
        Serializable.quick_init(self, locals())
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.use_cem = use_cem
        self.use_opt = use_opt
        self.use_opt_w_policy = use_opt_w_policy
        self.initializer_str = initializer_str
        self.kl_coef = kl_coef
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
        self.use_graph = use_graph

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1
        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"
        assert use_graph

        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_envs, self.obs_space_dims), name='obs')
        self.optimal_action = None
        if use_opt_w_policy:
            self.tau_optimizer = MPCTauOptimizer(max_epochs=self.num_opt_iters)
            self.policy = GaussianMLPPolicy(
                name='gaussian-mlp-policy',
                obs_dim=self.obs_space_dims,
                action_dim=self.action_space_dims,
                hidden_sizes=(64, 64),
                learn_std=True,
                hidden_nonlinearity=tf.tanh,  # TODO: tunable?
                output_nonlinearity=tf.tanh,  # TODO: scale to match action space range later
            )
            self.build_opt_graph_w_policy()
        elif use_opt:
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
            self.tau_optimizer = MPCTauOptimizer(max_epochs=self.num_opt_iters)
            self.build_opt_graph()
        elif use_cem:
            self.build_cem_graph()
        else:
            self.build_rs_graph()

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]

        return self.get_actions(observation)

    def get_actions(self, observations, do_grads_plots=False):
        sess = tf.get_default_session()
        if self.use_opt_w_policy:
            actions, tau_mean_val_0, tau_log_std_val_0, kl_0 = self.tau_optimizer.optimize({'obs': observations})
        elif self.use_opt:
            actions, tau_mean_val, tau_log_std_val_0, kl_0 = self.tau_optimizer.optimize(
                {'obs': observations, 'tau_mean': self.tau_mean_val},
                do_plots=do_grads_plots,
            )
            tau_mean_val_0 = tau_mean_val[0][0]
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
            actions, = sess.run([self.optimal_action], feed_dict={self.obs_ph: observations})
            return actions, []

        return actions, [dict(mean=tau_mean_val_0, std=np.exp(tau_log_std_val_0), kl=kl_0)]


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
            # dynamics_dist = self.dynamics_model.distribution_info_sym(observation, act[t])
            # mean, var = dynamics_dist['mean'], dynamics_dist['var']
            # next_observation = mean + tf.random.normal(shape=tf.shape(mean))*tf.sqrt(var)
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

    """
    def build_opt_graph(self):
        # Initialization
        returns = 0
        tau = tf.random.uniform(
            shape=[self.horizon, tf.shape(self.obs_ph)[0] * self.n_candidates, self.action_space_dims],
            minval=self.env.action_space.low,
            maxval=self.env.action_space.high,
        )

        # Equivalent to np.repeat
        observation = tf.reshape(
            tf.tile(tf.expand_dims(self.obs_ph, -1), [1, self.n_candidates, 1]),
            [-1, self.obs_space_dims]
        )
        observation = tf.stop_gradient(observation)

        for epoch in range(self.num_opt_iters):
            # compute return with current trajectories
            for t in range(self.horizon):
                next_observation = self.dynamics_model.predict_sym(observation, tau[t])
                next_observation = tf.stop_gradient(next_observation)
                assert self.reward_model is None
                rewards = self.unwrapped_env.tf_reward(observation, tau[t], next_observation)
                returns += self.discount ** t * rewards
                observation = next_observation

            # optimize trajectories
            grads_tau, = tf.gradients(ys=returns, xs=[tau,])
            tau += tf.multiply(self.opt_learning_rate, grads_tau)

        returns = tf.reshape(returns, (-1, self.n_candidates))  # (batch_size, n_candidates)

        cand_a = tf.reshape(tau[0], [-1, self.n_candidates, self.action_space_dims])  # (batch_size, n_candidates, act_dims)
        idx = tf.reshape(tf.argmax(returns, axis=1), [-1, 1])  # (batch_size, 1)
        self.optimal_action = tf.squeeze(tf.batch_gather(cand_a, idx), axis=1)

    def build_opt_graph(self):
        # Initialization
        returns = 0
        tau = tf.random.uniform(
            shape=[self.horizon, tf.shape(self.obs_ph)[0], self.action_space_dims],
            minval=self.env.action_space.low,
            maxval=self.env.action_space.high,
        )

        # observation = tf.stop_gradient(self.obs_ph)
        observation = self.obs_ph
        observation_seq = [observation]

        # compute return with current trajectories
        for t in range(self.horizon):
            next_observation = self.dynamics_model.predict_sym(observation, tau[t])
            # next_observation = tf.stop_gradient(next_observation)
            observation_seq.append(next_observation)
            assert self.reward_model is None
            rewards = self.unwrapped_env.tf_reward(observation, tau[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation

        grads_tau, = tf.gradients(ys=returns, xs=[tau,])
        tau += tf.multiply(self.opt_learning_rate, grads_tau)

        new_returns = 0
        for t in range(self.horizon):
            rewards = self.unwrapped_env.tf_reward(observation_seq[t], tau[t], observation_seq[t+1])
            new_returns += self.discount ** t * rewards

        self.optimal_action = tau[0]
        self.returns = returns
        self.new_returns = new_returns

    def opt_tau_sym(self, obs, tau):
        if self.model_params is None:
            self.model_params = sum([list(nn.get_params().values()) for nn in self.dynamics_model._networks], [])

        returns = 0
        for t in range(self.horizon):
            next_obs = self.dynamics_model.predict_sym(obs, tau[t])
            assert self.reward_model is None
            rewards = self.unwrapped_env.tf_reward(obs, tau[t], next_obs)
            returns += self.discount ** t * rewards
            obs = next_obs

        grads_tau, = tf.gradients(
            ys=returns,
            xs=[tau,],
            stop_gradients=self.model_params,
        )
        return tau + tf.multiply(self.opt_learning_rate, grads_tau), returns

    def build_opt_graph(self):
        # tau_mean = tf.get_variable(
        #     'tau_mean',
        #     shape=(self.num_envs, self.action_space_dims),
        #     dtype=tf.float32,
        #     trainable=True,
        # )
        # tau_std = tf.get_variable(
        #     'tau_std',
        #     shape=(self.num_envs, self.action_space_dims),
        #     dtype=tf.float32,
        #     trainable=True,
        # )
        # tau = tau_mean + tf.multiply(tf.random.normal((self.num_envs, self.action_space_dims)), tau_std)
        tau = tf.get_variable(
            'tau',
            shape=(self.horizon, self.num_envs, self.action_space_dims),
            dtype=tf.float32,
            initializer=tf.initializers.random_uniform(minval=self.env.action_space.low, maxval=self.env.action_space.high),
            trainable=True,
        )
        tf.assign(tau, tf.clip_by_value(tau, self.env.action_space.low, self.env.action_space.high))
        returns = tf.zeros(shape=(self.num_envs,))
        obs = self.obs_ph
        for t in range(self.horizon):
            next_obs = self.dynamics_model.predict_batches_sym(obs, tau[t])
            assert self.reward_model is None
            rewards = self.unwrapped_env.tf_reward(obs, tau[t], next_obs)
            returns += self.discount ** t * rewards
            obs = next_obs

        self.tau_optimizer.build_graph(
            loss=tf.reduce_mean(-returns, axis=0),
            var_list=[tau],
            # init_op=tf.assign(tau, tf.concat([
            #     tau[1:],
            #     tf.random_uniform(
            #         (1, self.num_envs, self.action_space_dims),
            #         minval=self.env.action_space.low,
            #         maxval=self.env.action_space.high,
            #     )
            # ], axis=0)),
            result_op=tau[0],
            input_ph_dict={'obs': self.obs_ph,},
        )
    """

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
        # log_std_var = tf.maximum(log_std_var, np.log(1e-6))
        old_mean_var = tf.get_variable(
            'old_tau_mean',
            shape=(self.horizon, self.num_envs, self.action_space_dims),
            dtype=tf.float32,
            trainable=False,
        )
        old_log_std_var = tf.get_variable(
            'old_tau_log_std',
            shape=(1, 1, self.action_space_dims),
            dtype=tf.float32,
            trainable=False,
        )
        init_ops = [
            tf.assign(mean_var, self.tau_mean_ph),
            tf.assign(old_mean_var, self.tau_mean_ph),
            tf.assign(old_log_std_var, log_std_var),
        ]
        tau = mean_var + tf.multiply(tf.random.normal(tf.shape(mean_var)), tf.exp(log_std_var))
        tau = tf.clip_by_value(tau, self.env.action_space.low, self.env.action_space.high)
        # TODO: rather than clipping, add penalty to actions outside valid range
        returns = tf.zeros(shape=(self.num_envs,))
        obs = self.obs_ph
        for t in range(self.horizon):
            next_obs = self.dynamics_model.predict_batches_sym(obs, tau[t])
            rewards = self.unwrapped_env.tf_reward(obs, tau[t], next_obs)
            returns += self.discount ** t * rewards
            obs = next_obs

        kl = tf.reduce_sum(DiagonalGaussian(-1).kl_sym(
            old_dist_info_vars=dict(
                mean=tf.reshape(old_mean_var, [-1, self.action_space_dims]),
                log_std=tf.reshape(old_log_std_var, [-1, self.action_space_dims]),
            ),
            new_dist_info_vars=dict(
                mean=tf.reshape(mean_var, [-1, self.action_space_dims]),
                log_std=tf.reshape(log_std_var, [-1, self.action_space_dims]),
            ),
        ), axis=0)/self.num_envs  # to match returns = reduce_sum(returns)/self.num_envs

        neg_returns = tf.reduce_mean(-returns, axis=0)

        self.tau_optimizer.build_graph(
            loss=neg_returns+self.kl_coef*kl,
            init_op=[init_ops],
            var_list=[mean_var, log_std_var],
            result_op=[tau[0], mean_var, log_std_var[0][0], kl],
            input_ph_dict={'obs': self.obs_ph, 'tau_mean': self.tau_mean_ph},
        )

    def build_opt_graph_w_policy(self):
        assert self.policy is not None
        returns = tf.zeros(shape=(self.num_envs,))
        obs = self.obs_ph
        old_mean_var = tf.get_variable(
            'old_policy_mean',
            shape=(self.num_envs, self.action_space_dims),
            dtype=tf.float32,
            trainable=False,
        )
        old_log_std_var = tf.get_variable(
            'old_policy_log_std',
            shape=(1, self.action_space_dims),
            dtype=tf.float32,
            trainable=False,
        )
        # lmbda = tf.get_variable(
        #     'lambda',
        #     shape=(),
        #     initializer=tf.initializers.ones,
        #     dtype=tf.float32,
        #     trainable=True,
        # )
        old_dist_policy = self.policy.distribution_info_sym(obs)
        init_op = [
            tf.assign(old_mean_var, old_dist_policy['mean']),
            tf.assign(old_log_std_var, old_dist_policy['log_std']),
        ]
        result_op = None
        p_info_dict = None
        for t in range(self.horizon):
            dist_policy = self.policy.distribution_info_sym(obs)
            act, dist_policy = self.policy.distribution.sample_sym(dist_policy)
            act = tf.clip_by_value(act, self.env.action_space.low, self.env.action_space.high)
            # TODO: penalize rather than clipping
            next_obs = self.dynamics_model.predict_batches_sym(obs, act)
            rewards = self.unwrapped_env.tf_reward(obs, act, next_obs)
            returns += self.discount ** t * rewards
            obs = next_obs
            if t == 0:
                result_op = [act, dist_policy['mean'][0], dist_policy['log_std'][0]]
                p_info_dict = dist_policy

        kl = tf.reduce_mean(self.policy.distribution.kl_sym(
            p_info_dict,
            dict(mean=old_mean_var, log_std=old_log_std_var),
        ), axis=0)

        neg_returns = tf.reduce_mean(-returns, axis=0)

        self.tau_optimizer.build_graph(
            loss=neg_returns+self.kl_coef*kl,
            var_list=list(self.policy.get_params().values()),
            init_op=init_op,
            result_op=result_op + [kl],
            input_ph_dict={'obs': self.obs_ph},
            #lmbda=lmbda,
            #loss_dual=tf.reduce_mean(-lmbda_kl, axis=0),
        )

    # def build_opt_graph_w_policy(self):
    #     assert self.policy is not None
    #     returns = tf.zeros(shape=(self.num_envs,))
    #     obs = self.obs_ph
    #     result_op = None
    #     for t in range(self.horizon):
    #         dist_policy = self.policy.distribution_info_sym(obs)
    #         act, dist_policy = self.policy.distribution.sample_sym(dist_policy)
    #         act = tf.clip_by_value(act, self.env.action_space.low, self.env.action_space.high)
    #         if t == 0:
    #             result_op = act
    #         next_obs = self.dynamics_model.predict_batches_sym(obs, act)
    #         assert self.reward_model is None
    #         rewards = self.unwrapped_env.tf_reward(obs, act, next_obs)
    #         returns += self.discount ** t * rewards
    #         obs = next_obs
    #
    #     self.tau_optimizer.build_graph(
    #         # loss=tf.reduce_mean(tf.multiply(sum_likelihood, tf.stop_gradient(-returns) + ), axis=0),
    #         loss=-tf.reduce_mean(returns), axis=0),
    #         var_list=list(self.policy.get_params().values()),
    #         result_op=result_op,
    #         input_ph_dict={'obs': self.obs_ph},
    #     )

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
