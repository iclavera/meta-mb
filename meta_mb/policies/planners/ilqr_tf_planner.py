from meta_mb.logger import logger
import scipy.linalg as sla
import numpy as np
import tensorflow as tf
import time
import copy
from meta_mb.policies import utils


class iLQRPlanner(object):
    def __init__(self, env, dynamics_model, num_envs, horizon, initializer_str,
                 num_ilqr_iters, reg_str='V', use_hessian_f=False, discount=1,
                 mu_min=1e-6, mu_max=1e10, mu_init=1e-5, delta_0=2, delta_init=1.0,
                 alpha_init=1.0, alpha_decay_factor=3.0,
                 c_1=1e-7, max_forward_iters=10, max_backward_iters=10,
                 n_candidates=1000, num_cem_iters=5, cem_deterministic_policy=True, cem_alpha=0.1, percent_elites=0.1):
        self._env = copy.deepcopy(env)
        self.dynamics_model = dynamics_model
        self.num_envs = num_envs
        self.horizon = horizon
        self.initializer_str = initializer_str
        self.num_ilqr_iters = num_ilqr_iters
        self.reg_str = reg_str
        self.use_hessian_f = use_hessian_f
        self.discount = discount
        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_init = mu_init
        self.alpha_init = alpha_init
        self.alpha_decay_factor = alpha_decay_factor
        self.delta_0 = delta_0
        self.delta_init = delta_init
        self.c_1 = c_1
        self.max_forward_iters = max_forward_iters
        self.max_backward_iters = max_backward_iters

        # cem
        self.n_candidates = n_candidates
        self.num_cem_iters = num_cem_iters
        self.cem_deterministic_policy = cem_deterministic_policy
        self.cem_alpha = cem_alpha
        self.num_elites = int(percent_elites * n_candidates)

        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        self.param_array = None

        self.u_ph_ha = tf.placeholder(
            dtype=tf.float32, shape=(self.horizon, self.act_dim), name='u',
        )
        self.x_ph_o = tf.placeholder(
            dtype=tf.float32, shape=(self.obs_dim,), name='x',
        )

        build_time = time.time()
        self.opt_disc_reward_var, self.opt_u_var_ha = self._build_action_graph()
        self.cem_optimized_actions = self._build_cem_graph()
        logger.log('TimeBuildGraph', build_time - time.time())

    def _build_cem_graph(self):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        horizon = self.horizon
        n_candidates = self.n_candidates
        num_envs = self.num_envs
        alpha = self.cem_alpha

        mean = tf.ones(shape=[horizon, num_envs, 1, act_dim]) * (self.act_high + self.act_low) / 2
        var = tf.ones(shape=[horizon, num_envs, 1, act_dim]) * (self.act_high - self.act_low) / 16

        init_obs = tf.reshape(
            tf.tile(tf.expand_dims(self.x_ph_o, axis=1), [1, n_candidates, 1]),
            shape=(num_envs*n_candidates, obs_dim),
        )

        for itr in range(self.num_cem_iters):
            lb_dist, ub_dist = mean - self.act_low, self.act_high - mean
            constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
            std = tf.sqrt(constrained_var)
            act = mean + tf.random.normal(shape=[horizon, num_envs, n_candidates, act_dim]) * std
            act = tf.clip_by_value(act, self.act_low, self.act_high)
            act = tf.reshape(act, shape=(horizon, num_envs*n_candidates, act_dim))
            returns = tf.zeros((num_envs*n_candidates,))

            obs = init_obs
            for t in range(horizon):
                next_obs = self.dynamics_model.predict_sym(obs, act[t])
                rewards = self._env.tf_reward(obs, act[t], next_obs)
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
            mean = mean * alpha + elite_mean * (1 - alpha)
            var = var * alpha + elite_var * (1 - alpha)

        optimized_actions_mean = mean[:, :, 0, :]
        if self.cem_deterministic_policy:
            return optimized_actions_mean
        else:
            optimized_actions_var = var[:, :, 0, :]
            return optimized_actions_mean + \
                   tf.random.normal(shape=tf.shape(optimized_actions_mean)) * tf.sqrt(optimized_actions_var)

    def _build_action_graph(self):
        """

        :return: build action graph for one environment
        """
        u_ha = self.u_ph_ha
        num_models = self.dynamics_model.num_models
        horizon = self.horizon
        obs_dim, act_dim = self.obs_dim, self.act_dim

        # compute gradients, collect x_ho and disc_reward
        x_ho = tf.TensorArray(
            dtype=tf.float32, size=(horizon,), clear_after_read=False, tensor_array_name='obs_ho',
        )
        gradients_list = [tf.TensorArray(
            dtype=tf.float32, size=(horizon,), clear_after_read=True, tensor_array_name='gradients',
        ) for _ in range(10)]
        x = self.x_ph_o
        disc_reward = 0

        for i in range(horizon):
            if self.use_hessian_f:
                u = u_ha[i, :]
                x_u_concat = tf.concat([x, u], axis=0)
                x, u = x_u_concat[:obs_dim], x_u_concat[-act_dim:]
                x_prime = self.dynamics_model.predict_sym(x[None], u[None], pred_type=model_idx[i])[0]

                hess = utils.hessian_wrapper(x_prime, x_u_concat, obs_dim, obs_dim+act_dim)
                f_x = utils.jacobian_wrapper(x_prime, x, obs_dim, obs_dim)
                f_u = utils.jacobian_wrapper(x_prime, u, obs_dim, act_dim)
                f_xx = hess[:, :obs_dim, :obs_dim]
                f_uu = hess[:, -act_dim:, -act_dim:]
                f_ux = (hess[:, -act_dim:, :obs_dim] + tf.transpose(hess[:, :obs_dim, -act_dim:], perm=[0, 2, 1])) * 0.5
                df = [f_x, f_u, f_xx, f_uu, f_ux]

            else:
                u = u_ha[i, :]
                x_prime = self.dynamics_model.predict_sym(x[None], u[None], pred_type=model_idx[i])[0]

                # compute gradients
                f_x = utils.jacobian_wrapper(x_prime, x, obs_dim, obs_dim)
                f_u = utils.jacobian_wrapper(x_prime, u, obs_dim, act_dim)
                df = [f_x, f_u, None, None, None]

            # store
            dl = self._env.tf_dl(x, u, x_prime)
            for grad_idx, grad in enumerate(df + dl):
                if grad is not None:
                    gradients_list[grad_idx].write(i, grad)
            reward = self._env.tf_reward(x, u, x_prime)
            disc_reward += self.discount**i * reward
            x_ho.append(x)
            x = x_prime
        J_val = -disc_reward

        # backward pass
        def body(accept, i, open_k_array, closed_K_array, delta_J_1, delta_J_2, V_prime_x, V_prime_xx):
            f_x, f_u, f_xx, f_uu, f_ux, l_x, l_u, l_xx, l_uu, l_ux = [gradients_list[grad_idx].read(i) for grad_idx in range(10)]
            Q_x = l_x + tf.transpose(f_x) @ V_prime_x
            Q_u = l_u + tf.transpose(f_u) @ V_prime_x
            if self.use_hessian_f:
                Q_xx = l_xx + tf.transpose(f_x) @ V_prime_xx @ f_x + tf.tensordot(V_prime_x, f_xx, axes=1)
                Q_uu = l_uu + tf.transpose(f_u) @ V_prime_xx @ f_u + tf.tensordot(V_prime_x, f_uu, axes=1)
                Q_uu = (Q_uu + tf.transpose(Q_uu)) * 0.5
                Q_ux = l_ux + tf.transpose(f_u) @ V_prime_xx @ f_x + tf.tensordot(V_prime_x, f_ux, axes=1)
            else:
                Q_xx = l_xx + tf.transpose(f_x) @ V_prime_xx @ f_x
                Q_uu = l_uu + tf.transpose(f_u) @ V_prime_xx @ f_u
                Q_ux = l_ux + tf.transpose(f_u) @ V_prime_xx @ f_x

            # use conjugate gradient method to solve for
            accept_k, k = utils.tf_cg(f_Ax=lambda k: Q_uu @ k, b=-Q_u, cg_iters=self.cg_iters, residual_tol=1e-10)
            accept_K, K = utils.tf_cg(f_Ax=lambda K: Q_uu @ K, b=-Q_ux, cg_iters=self.cg_iters, residual_tol=1e-10)  # TODO: b=-Q_ux_reg?
            new_accept = tf.logical_or(accept_k, accept_K)

            open_k_array.write(i, k)
            closed_K_array.write(i, K)
            delta_J_1 += k.T @ Q_u
            delta_J_2 += k.T @ Q_uu @ k

            # prepare for next i
            i -= 1
            V_prime_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
            V_prime_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            V_prime_xx = (V_prime_xx + V_prime_xx.T) * 0.5

            return (new_accept, i, open_k_array, closed_K_array, V_prime_x, V_prime_xx, delta_J_1, delta_J_2)

        # define variables for backward while loop
        i = tf.Variable(initial_value=horizon-1, trainable=False, dtype=tf.int32)
        V_prime_xx, V_prime_x = tf.zeros((obs_dim, obs_dim)), tf.zeros((obs_dim,))
        open_k_array = tf.TensorArray(
            dtype=tf.float32, size=(horizon,), element_shape=(act_dim,), clear_after_read=False, tensor_array_name='open_k',
        )
        closed_K_array = tf.TensorArray(
            dtype=tf.float32, size=(horizon,), element_shape=(act_dim, obs_dim), clear_after_read=False, tensor_array_name='closed_K',
        )
        delta_J_1 = tf.Variable(initial_value=0, trainable=False, name="delta_J_1")
        delta_J_2 = tf.Variable(initial_value=0, trainable=False, name="delta_J_2")
        accept = tf.constant(True)

        # if no error: do the i-th step of backward pass, i = horizon-1, ..., 0
        cond = lambda accept, *args: accept
        loop_vars = [accept, i, open_k_array, closed_K_array, delta_J_1, delta_J_2, V_prime_x, V_prime_xx]
        backward_accept, _, open_k_array, closed_K_array, delta_J_1, delta_J_2, _, _ = tf.while_loop(cond=cond, body=body, loop_vars=loop_vars, maximum_iterations=horizon)

        # forward pass if backward accept is True
        def body(accept, alpha, opt_J_val, opt_u_ha):
            opt_u_ha = []
            x = self.x_ph_o
            delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
            disc_reward = tf.Variable(initial_value=0, trainable=False, name='opt_J_val_curr_model')
            for i in range(horizon):
                u = u_ha[i] + alpha * open_k_array.read(i) + closed_K_array.read(i) @ (x - x_ho[i])
                u = tf.clip_by_value(u, self.act_low, self.act_high)
                opt_u_ha.append(u)

                x_prime = self.dynamics_model.predict_dym(x[None], u[None], pred_type=tf.random.uniform(shape=(), maxval=num_models, dtype=tf.int32))[0]
                reward = self._env.tf_reward(x, u, x_prime)
                disc_reward += self.discount**i * reward
                x = x_prime
            opt_J_val = -disc_reward

            accept = tf.math.logical_and(tf.greater(J_val, opt_J_val), tf.greater(J_val-opt_J_val, self.c_1*delta_J_alpha))
            next_alpha = alpha / self.alpha_decay_factor
            return (accept, next_alpha, opt_J_val, opt_u_ha)

        accept = tf.constant(False)
        alpha = self.alpha_init
        opt_J_val = J_val
        opt_u_ha = u_ha

        # if not accepted: do full forward pass with alpha
        cond = lambda accept, *args: tf.logical_and(tf.logical_not(accept), backward_accept)
        loop_vars = [accept, alpha, opt_J_val, opt_u_ha]
        _, _, opt_J_val, opt_u_ha = tf.while_loop(cond=cond, body=body, loop_vars=loop_vars, maximum_iterations=self.max_backward_iters)

        return -opt_J_val, opt_u_ha

    def get_actions(self, obs, verbose=True):
        if self.u_array_val is None:
            self._reset(obs)
        self._reset_mu()
        sess = tf.get_default_session()
        feed_dict = {self.u_array_ph: self.u_array_val, self.obs_ph: obs, self.model_idx_ph: self.model_idx_val}  # self.u_array_val is updated over iterations

        active_envs = list(range(self.num_envs))
        for itr in range(self.num_ilqr_iters):
            # self.u_array_val is changed along optimization
            # compute: x_array, J_val, f_x, f_u, f_xx, f_uu, f_ux, l_x, l_u, l_xx, l_uu, l_ux
            utils_by_env = sess.run(self.utils_sym_by_env, feed_dict=feed_dict)

            # if itr == 0 and verbose:
            #     for env_idx in range(self.num_envs):
            #         logger.logkv(f'RetDyn-{env_idx}', -utils_by_env[env_idx][2])
            #     logger.dumpkvs()

            for env_idx in active_envs.copy():
                try:
                    # assert np.allclose(utils_by_env[env_idx][0][0], obs[env_idx])  # compare x_array[0] and obs for the current environment
                    success, agent_info = self.update_u_per_env(env_idx, utils_by_env[env_idx])
                    if verbose:
                        logger.logkv(f'RetDyn-{env_idx}', agent_info['returns'])
                        logger.logkv(f'RetEnv-{env_idx}', agent_info['gt_returns'])
                        if success:
                            logger.logkv(f'u_clipped_pct-{env_idx}', agent_info['u_clipped_pct'])
                            logger.logkv('alpha', agent_info['alpha'])
                            logger.logkv('ensemble_ratio', agent_info['ensemble_ratio'])
                            logger.logkv('eff_ratio', agent_info['eff_ratio'])
                            logger.logkv('gt_ratio', agent_info['gt_ratio'])
                            logger.logkv('gt_diff', agent_info['gt_diff'])

                    # if success and agent_info['u_clipped_pct'] > 0.85:  # actions all on boundary
                    #     self.u_array_val[:, env_idx, :] = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.horizon, self.act_dim))

                    # # sanity check  # FIXME: assertion fails with ensemble or stochastic model
                    # tf_opt_J_val = sess.run(self.utils_sym_by_env[env_idx][2], feed_dict=feed_dict)
                    # logger.log(f"itr = {itr}, idx = {env_idx}, opt_J_val = {-agent_info['returns']}, tf_opt_J_val = {tf_opt_J_val}")
                    # tf_opt_J_val = sess.run(self.utils_sym_by_env[env_idx][2], feed_dict=feed_dict)
                    # logger.log(f"itr = {itr}, idx = {env_idx}, opt_J_val = {-agent_info['returns']}, tf_opt_J_val = {tf_opt_J_val}")
                    # assert tf_opt_J_val == -agent_info['returns']
                except OverflowError:
                    logger.log(f'{env_idx}-th env is removed at iLQR_itr {itr}')
                    active_envs.remove(env_idx)

            if len(active_envs) == 0:
                break

        if verbose:
            logger.dumpkvs()

        # logging: RetEnv for env_idx = 0
        # if np.random.rand() < 0.3 and verbose:  # hack to prevent: Got MuJoCo Warning: Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable. Time = 0.4400.
        #     for env_idx in range(self.num_envs):
        #         logger.logkv(f'RetEnv-{env_idx}', self._run_open_loop(self.u_array_val[:, env_idx, :], obs[env_idx, :]))  # USE RetEnv HERE TO MEASURE PERFORMANCE
        #     logger.dumpkvs()

        optimized_actions = self.u_array_val.copy()  # (horizon, num_envs, act_dim)

        # shift u_array and perm
        self._shift()  # FIXME: alternative ways to initialize u_new

        return optimized_actions, []#, next_obs

    def _gt_J_val(self, init_obs, u_array):
        _ = self._env.reset_from_obs(init_obs)
        disc_reward = 0
        for i in range(self.horizon):
            _, reward, _, _ = self._env.step(u_array[i])
            disc_reward += self.discount**i * reward
        return disc_reward

    def _shift(self):
        #  shift u_array
        if self.initializer_str == 'cem':
            self.u_array_val = None
            self.model_idx_val = None
        else:
            # u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.num_envs,self.act_dim))
            # u_new = np.mean(self.u_array_val, axis=0) + np.random.normal(size=(self.num_envs, self.act_dim)) * np.std(self.u_array_val, axis=0)
            if self.initializer_str == 'uniform':
                u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.num_envs, self.act_dim))
            elif self.initializer_str == 'zeros':
                u_new = np.random.normal(scale=0.1, size=(self.num_envs, self.act_dim))
                u_new = np.clip(u_new, a_min=self.act_low, a_max=self.act_high)
            else:
                raise RuntimeError
            self.u_array_val = np.concatenate([self.u_array_val[1:, :, :], u_new[None]])
            self.model_idx_val = np.concatenate([self.model_idx_val[1:, :], np.random.randint(self.dynamics_model.num_models, size=(1, self.num_envs))])

    def _reset(self, obs):
        if self.initializer_str == 'cem':
            logger.log('initialize with cem...')
            sess = tf.get_default_session()
            init_u_array, = sess.run([self.cem_optimized_actions], feed_dict={self.obs_ph: obs})
        elif self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.act_low, high=self.act_high,
                                             size=(self.horizon, self.num_envs, self.act_dim))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.num_envs, self.act_dim))
            init_u_array = np.clip(init_u_array, a_min=self.act_low, a_max=self.act_high)
        else:
            raise RuntimeError

        self.u_array_val = init_u_array
        self.model_idx_val = np.random.randint(low=0, high=self.dynamics_model.num_models, size=(self.horizon, self.num_envs))

    def warm_reset(self, u_array):
        if self.initializer_str != 'cem':
            self.u_array_val = u_array
            self.model_idx_val = self.model_idx_val_init
        else:
            self.u_array_val = None
            self.model_idx_val = None


def chol_inv(matrix):
    """
    Copied from mbbl.
    :param matrix:
    :return:
    """
    L = np.linalg.cholesky(matrix)
    L_inv = sla.solve_triangular(
        L, np.eye(len(L)), lower=True, check_finite=False
    )
    matrix_inv = L_inv.T.dot(L_inv)
    return matrix_inv
