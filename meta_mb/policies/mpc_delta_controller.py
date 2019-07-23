from meta_mb.utils.serializable import Serializable
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.policies.distributions.diagonal_gaussian import DiagonalGaussian
from meta_mb.optimizers.mpc_tau_optimizer import MPCTauOptimizer
import tensorflow as tf
import numpy as np


class MPCDeltaController(Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            num_rollouts=None,
            reward_model=None,
            discount=1,
            use_opt_w_policy=False,
            initializer_str='uniform',
            reg_coef=1,
            horizon=10,
            num_opt_iters=8,
            opt_learning_rate=1e-3,
            percent_elites=0.1,
            alpha=0.1,
            num_particles=20,
    ):
        Serializable.quick_init(self, locals())
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.use_opt_w_policy = use_opt_w_policy
        self.initializer_str = initializer_str
        self.reg_coef = reg_coef
        self.horizon = horizon
        self.num_opt_iters = num_opt_iters
        self.opt_learning_rate = opt_learning_rate
        self.num_envs = num_rollouts
        self.percent_elites = percent_elites
        self.env = env
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

        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_envs, self.obs_space_dims), name='obs')
        """
        self.optimal_action always contain actions executed throughout the sampled trajectories. 
        It is the first row of tau, where tau is used for looking-ahead planning. 
        """
        self.optimal_action = None
        if use_opt_w_policy:
            self.deltas_optimizer = MPCTauOptimizer(max_epochs=self.num_opt_iters)
            self.delta_policy = GaussianMLPPolicy(
                name='gaussian-mlp-policy',
                obs_dim=self.obs_space_dims,
                action_dim=self.action_space_dims,
                hidden_sizes=(64, 64),
                learn_std=True,
                hidden_nonlinearity=tf.tanh,  # TODO: tunable?
                output_nonlinearity=tf.tanh,  # TODO: scale to match action space range later
            )
            self.build_opt_graph_w_policy()
        else:
            self.deltas_mean_val = np.zeros(
                (self.horizon, self.num_envs, self.action_space_dims),
            )
            self.deltas_mean_ph = tf.placeholder(
                dtype=tf.float32,
                shape=np.shape(self.deltas_mean_val),
                name='deltas_mean',
            )
            self.deltas_optimizer = MPCTauOptimizer(max_epochs=self.num_opt_iters)
            self.build_opt_graph()

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]

        return self.get_actions(observation)

    def get_actions(self, observations, return_first_info=False):
        agent_infos = []

        if self.use_opt_w_policy:
            actions, act_mean_val_0,  = self.deltas_optimizer.optimize({'obs': observations})
        else:
            # info to plot action executed in the fist env (observation)
            if return_first_info:
                sess = tf.get_default_session()
                prev_action = sess.run(self.optimal_action[0])

            actions, deltas_mean_val, neg_returns, reg = self.deltas_optimizer.optimize(
                {'obs': observations, 'deltas_mean': self.deltas_mean_val},
            )

            if return_first_info:
                mean_val_0 = deltas_mean_val[0]
                log_std_val = sess.run(self.log_std_var)
                agent_infos = [dict(mean=prev_action + mean_val_0, std=np.exp(log_std_val))]

            # rotate
            self.deltas_mean_val = np.concatenate([
                deltas_mean_val[1:],
                np.zeros((1, self.num_envs, self.action_space_dims)),
            ], axis=0)

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

    def build_opt_graph(self):
        self.optimal_action = tf.get_variable(
            'optimal_action',
            shape=(self.num_envs, self.action_space_dims),
            dtype=tf.float32,
            trainable=False,
        )
        mean_var = tf.get_variable(
            'deltas_mean',
            shape=(self.horizon, self.num_envs, self.action_space_dims),
            dtype=tf.float32,
            trainable=True,
        )
        log_std_var = tf.get_variable(
            'deltas_log_std',
            shape=(1, 1, self.action_space_dims),
            dtype=tf.float32,
            initializer=tf.initializers.ones,
            trainable=True,
        )
        # log_std_var = tf.maximum(log_std_var, np.log(1e-6))
        deltas = mean_var + tf.multiply(tf.random.normal(tf.shape(mean_var)), tf.exp(log_std_var))
        acts, obs = self.optimal_action, self.obs_ph
        returns = tf.zeros(shape=(self.num_envs,))
        for t in range(self.horizon):
            acts = acts + deltas[t]
            acts = tf.clip_by_value(acts, self.env.action_space.low, self.env.action_space.high)
            # TODO: rather than clipping, add penalty to actions outside valid range
            next_obs = self.dynamics_model.predict_batches_sym(obs, acts)
            rewards = self.unwrapped_env.tf_reward(obs, acts, next_obs)
            returns += self.discount ** t * rewards
            obs = next_obs

        neg_returns = tf.reduce_mean(-returns, axis=0)
        reg = tf.norm(deltas)/self.num_envs

        self.deltas_optimizer.build_graph(
            loss=neg_returns+self.reg_coef*reg,
            init_op=[tf.assign(mean_var, self.deltas_mean_ph)],
            var_list=[mean_var, log_std_var],
            result_op=[tf.assign_add(self.optimal_action, deltas[0]), mean_var, neg_returns, reg],
            input_ph_dict={'obs': self.obs_ph, 'deltas_mean': self.deltas_mean_ph},
        )

        self.mean_var, self.log_std_var = mean_var, log_std_var

    def build_opt_graph_w_policy(self):
        assert self.policy is not None
        returns = tf.zeros(shape=(self.num_envs,))
        obs = self.obs_ph
        for t in range(self.horizon):
            dist_policy = self.delta_policy.distribution_info_sym(obs)
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
