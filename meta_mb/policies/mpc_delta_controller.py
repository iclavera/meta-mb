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
            reg_str=None,
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
        self.reg_str = reg_str
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
        # if use_opt_w_policy:
        #     self.deltas_optimizer = MPCTauOptimizer(max_epochs=self.num_opt_iters)
        #     self.delta_policy = GaussianMLPPolicy(
        #         name='gaussian-mlp-policy',
        #         obs_dim=self.obs_space_dims,
        #         action_dim=self.action_space_dims,
        #         hidden_sizes=(64, 64),
        #         learn_std=True,
        #         hidden_nonlinearity=tf.tanh,  # TODO: tunable?
        #         output_nonlinearity=tf.tanh,  # TODO: scale to match action space range later
        #     )
        #     self.build_opt_graph_w_policy()

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

    def get_actions(self, observations, return_first_info=False, log_global_norms=False):
        # info to plot action executed in the fist env (observation)
        result = self.deltas_optimizer.optimize(
            {'obs': observations, 'deltas_mean': self.deltas_mean_val},
            run_extra_result_op=return_first_info,
            log_global_norms=log_global_norms,
        )
        if return_first_info:
            actions, deltas_mean_val, neg_returns, reg, mean, std = result
            agent_infos = [dict(mean=mean, std=std, reg=reg)]
        else:
            actions, deltas_mean_val, neg_returns, reg = result
            agent_infos = []

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
        prev_actions = tf.get_variable(
            'prev_actions',
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

        optimal_actions_mean = prev_actions + mean_var[0]
        optimal_actions_var = tf.exp(log_std_var[0])
        # log_std_var = tf.maximum(log_std_var, np.log(1e-6))

        deltas = mean_var + tf.multiply(tf.random.normal(tf.shape(mean_var)), tf.exp(log_std_var))
        acts, obs = prev_actions, self.obs_ph
        returns, reg = tf.zeros(shape=(self.num_envs,)), tf.zeros(shape=(self.num_envs,))
        for t in range(self.horizon):
            acts = acts + deltas[t]
            acts = tf.clip_by_value(acts, self.env.action_space.low, self.env.action_space.high)
            # TODO: rather than clipping, add penalty to actions outside valid range

            if self.reg_str == 'uncertainty':
                pred_obs = self.dynamics_model.predict_sym_all(obs, acts)  # (num_envs, obs_space_dims, num_models)
                uncertainty = tf.math.reduce_variance(pred_obs, axis=-1)
                uncertainty = tf.reduce_sum(uncertainty, axis=1)
                reg += uncertainty
                # batch_gather params = (num_envs, num_models, obs_space_dims), indices = (num_envs, 1)
                idx = tf.random.uniform(shape=(self.num_envs,), minval=0, maxval=self.dynamics_model.num_models, dtype=tf.int32)
                next_obs = tf.batch_gather(tf.transpose(pred_obs, (0, 2, 1)), tf.reshape(idx, [-1, 1]))
                next_obs = tf.squeeze(next_obs, axis=1)
            else:
                next_obs = self.dynamics_model.predict_batches_sym(obs, acts)

            rewards = self.unwrapped_env.tf_reward(obs, acts, next_obs)
            returns += self.discount ** t * rewards
            obs = next_obs

        # build loss = total_cost + regularization
        neg_returns = tf.reduce_mean(-returns, axis=0)

        if self.reg_str == 'norm':
            reg = tf.norm(deltas) / self.num_envs
        elif self.reg_str == 'uncertainty' or self.reg_str is None:
            reg = tf.reduce_mean(reg, axis=0)
        else:
            raise NotImplementedError

        with tf.control_dependencies([optimal_actions_mean]):
            result_op = [tf.assign_add(prev_actions, deltas[0]), mean_var, neg_returns, reg]
        extra_result_op = [optimal_actions_mean[0], optimal_actions_var[0]]

        self.deltas_optimizer.build_graph(
            loss=neg_returns+self.reg_coef*reg,
            init_op=[tf.assign(mean_var, self.deltas_mean_ph)],
            var_list=[mean_var, log_std_var],
            result_op=result_op,
            extra_result_op=extra_result_op,
            input_ph_dict={'obs': self.obs_ph, 'deltas_mean': self.deltas_mean_ph},
        )

    def plot_global_norms(self):
        self.deltas_optimizer.plot_global_norms()

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
