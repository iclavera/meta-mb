from meta_mb.logger import logger
import scipy.linalg as sla
import numpy as np
from collections import OrderedDict
import tensorflow as tf
import itertools


class DyniLQRPlanner(object):
    """
    Compute ground truth derivatives.
    """
    def __init__(self, env, dynamics_model, num_envs, horizon, u_array, num_ilqr_iters ,reg_str='V', discount=1,
                 mu_min=1e-6, mu_max=1e10, mu_init=1e-5, delta_0=2, delta_init=1.0, alpha_decay_factor=3.0,
                 c_1=1e-7, max_forward_iters=10, max_backward_iters=10,
                 forward_stop_cond='rel',
                 use_hessian_f=False, verbose=False):
        self._env = env
        self.dynamics_model = dynamics_model
        self.num_envs = num_envs
        self.horizon = horizon
        self.u_array_val = u_array
        self.num_ilqr_iters = num_ilqr_iters
        self.reg_str = reg_str
        self.discount = discount
        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_init = mu_init
        self.alpha_decay_factor = alpha_decay_factor
        self.delta_0 = delta_0
        self.delta_init = delta_init
        self.c_1 = c_1
        self.max_forward_iters = max_forward_iters
        self.max_backward_iters = max_backward_iters
        self.forward_stop_cond = forward_stop_cond
        self.use_hessian_f = use_hessian_f
        self.verbose = verbose
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        self._reset_mu()

        self.u_array_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(self.horizon, self.num_envs, self.act_dim),
            name='u_array',
        )
        self.obs_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(self.num_envs, self.obs_dim),
            name='obs',
        )

        self.utils_sym_by_env = self._build_deriv_graph()

    def _build_deriv_graph(self):
        x_array, df_array, dl_array = [], [], []

        obs = self.obs_ph
        returns = tf.zeros((self.num_envs,))
        for i in range(self.horizon):
            x_array.append(obs)
            acts = self.u_array_ph[i, :, :]
            next_obs = self.dynamics_model.predict_sym(obs, acts)

            df = self._df_sym(obs, acts, next_obs)
            df_array.append(df)
            dl = self._env.tf_dl_dict(obs, acts, next_obs, self.num_envs).values()
            dl_array.append(dl)
            rewards = self._env.tf_reward(obs, acts, next_obs)
            returns += self.discount**i * rewards

            obs = next_obs

        x_array = tf.stack(x_array, axis=0)
        J_val = -returns
        x_array_by_env = [x_array[:, env_idx] for env_idx in range(self.num_envs)]
        J_val_by_env = [J_val[env_idx] for env_idx in range(self.num_envs)]

        # outputs by deriv type: (horizon, num_envs, *)
        f_array = list(map(lambda arr: tf.stack(arr, axis=0), zip(*df_array)))
        l_array = list(map(lambda arr: tf.stack(arr, axis=0), zip(*dl_array)))

        # store by env
        derivs_by_env = list(map(lambda arr: [arr[:, env_idx] for env_idx in range(self.num_envs)], f_array + l_array))

        utils_sym_by_env = list(zip(x_array_by_env, J_val_by_env, *derivs_by_env))
        return utils_sym_by_env

    def _gradients_wrapper(self, y, x, dim_y):
        """

        :param y: (num_envs, dim_y)
        :param x: (num_envs, dim_x)
        :param dim_y:
        :param dim_x:
        :return: (num_envs, dim_y, dim_x)
        """
        jac_array = []
        for i in range(dim_y):
            jac_array.extend(tf.gradients(ys=y[:, i], xs=[x]))
        return tf.stack(jac_array, axis=1)

    def _df_sym(self, obs, acts, next_obs):
        """

        :param obs: (num_envs, obs_dim)
        :param acts: (num_envs, act_dim)
        :param next_obs: (num_envs, obs_dim)
        """
        # FIXME: ???
        # obs = tf.stop_gradient(obs)
        # acts = tf.stop_gradient(acts)

        # jac_f_x: (num_envs, obs_dim, obs_dim)
        jac_f_x = self._gradients_wrapper(next_obs, obs, self.obs_dim)

        # jac_f_u: (num_envs, obs_dim, act_dim)
        jac_f_u = self._gradients_wrapper(next_obs, acts, self.obs_dim)

        return jac_f_x, jac_f_u
    #
    # def update_x_u_for_one_step(self, obs):
    #     """
    #
    #     :param obs: (num_envs, obs_dim)
    #     :return:
    #     """
    #     u_array = self.u_array_val
    #     optimized_action = u_array[0, :, :]  # (num_envs, act_dim)
    #     for info_dict in self.info_dict_array:
    #         info_dict.update(dict(backward_accept=False, forward_accept=False))
    #     active_env_idx = np.arange(self.num_envs)
    #     num_active_envs = self.num_envs
    #
    #     """
    #     Derivatives
    #     """
    #     feed_dict = {self.u_array_ph: self.u_array_val, self.obs_ph: obs}
    #     sess = tf.get_default_session()
    #     global_x_array, global_J_val, global_f_x, global_f_u, global_l_x, global_l_u, global_l_xx, global_l_uu, global_l_ux = sess.run(
    #         list(self.utils_sym_dict.values()),
    #         feed_dict=feed_dict,
    #     )
    #     """
    #     Backward Pass
    #     """
    #     backward_pass_counter = 0
    #     global_open_k_array, global_closed_K_array = np.empty((self.horizon, self.num_envs)), np.empty((self.horizon, self.num_envs,))
    #     global_delta_J_1, global_delta_J_2 = np.zeros((self.num_envs,)), np.zeros((self.num_envs,))
    #     while num_active_envs > 0 and backward_pass_counter < self.max_backward_iters:  # and self.mu <= self.mu_max:
    #         # initialize
    #         global_open_k_array[active_env_idx] = None
    #         global_closed_K_array[active_env_idx] = None
    #         global_delta_J_1[active_env_idx] = 0
    #         global_delta_J_2[active_env_idx] = 0
    #
    #         f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux = (
    #             global_f_x[:, active_env_idx],
    #             global_f_u[:, active_env_idx],
    #             global_l_x[:, active_env_idx],
    #             global_l_u[:, active_env_idx],
    #             global_l_xx[:, active_env_idx],
    #             global_l_uu[:, active_env_idx],
    #             global_l_ux[:, active_env_idx],
    #         )
    #         V_prime_xx, V_prime_x = np.zeros((num_active_envs, self.obs_dim, self.obs_dim)), np.zeros((num_active_envs, self.obs_dim))  # l_x[-1], l_xx[-1]
    #
    #         try:
    #             # backward pass
    #             for i in range(self.horizon-1, -1, -1):
    #                 # compute Q
    #                 f_u_i_T = np.transpose(f_u[i], axes=[0, 2, 1])  # (num_active_envs, obs_dim, obs_dim)
    #                 Q_uu = l_uu[i] + f_u_i_T @ V_prime_xx @ f_u[i]  # (num_active_envs, act_dim, act_dim)
    #                 if self.reg_str == 'Q':
    #                     Q_uu_reg = Q_uu + self.mu * np.eye(self.act_dim)
    #                 elif self.reg_str == 'V':
    #                     Q_uu_reg = Q_uu + self.mu * f_u_i_T @ f_u[i]
    #                 else:
    #                     raise NotImplementedError
    #
    #                 if not np.allclose(Q_uu, np.transpose(Q_uu, axes=[0, 2, 1])):
    #                     print(Q_uu)
    #                     raise RuntimeError
    #
    #                 Q_uu_reg_inv = chol_inv(Q_uu_reg)  # except error here  # (num_active_envs, act_dim, act_dim)
    #
    #                 f_x_i_T = np.transpose(f_x[i], axes=[0, 2, 1])  # (num_active_envs, obs_dim, act_dim)
    #                 V_prime_x_expand = np.expand_dims(V_prime_x, axis=2)  # (num_active_envs, obs_dim, 1)
    #                 Q_x = l_x[i] + (f_x_i_T @ V_prime_x_expand)[:, :, 0]  # (num_active_envs, obs_dim)
    #                 Q_u = l_u[i] + (f_u_i_T @ V_prime_x_expand)[:, :, 0]  # (num_active_envs, act_dim)
    #                 Q_xx = l_xx[i] + f_x_i_T @ V_prime_xx @ f_x[i]  # (num_active_envs, obs_dim, obs_dim)
    #                 Q_ux = l_ux[i] + f_u_i_T @ V_prime_xx @ f_x[i]  # (num_active_envs, act_dim, obs_dim)
    #                 if self.reg_str == 'Q':
    #                     Q_ux_reg = Q_ux
    #                 elif self.reg_str == 'V':
    #                     Q_ux_reg = Q_ux + self.mu * f_u_i_T @ f_x[i]
    #                 else:
    #                     raise NotImplementedError
    #
    #                 # compute control matrices
    #                 # Q_uu_inv = np.linalg.inv(Q_uu)
    #                 Q_u_expand = np.expand_dims(Q_u, axis=2)  # (num_active_envs, act_dim, 1)
    #                 k = - (Q_uu_reg_inv @ Q_u_expand)[:, :, 0]  # (num_active_envs, act_dim)
    #                 K = - Q_uu_reg_inv @ Q_ux_reg  # (num_active_envs, act_dim, obs_dim)
    #                 global_open_k_array[i, active_env_idx] = k
    #                 global_closed_K_array[i, active_env_idx] = K
    #
    #                 k_expand = np.expand_dims(k, axis=2)  # (num_active_envs, act_dim, 1)
    #                 k_expand_T = np.transpose(k_expand, axes=[0, 2, 1])  # (num_active_envs, 1, act_dim)
    #                 K_T = np.transpose(K, axes=[0, 2, 1])  # (num_active_envs, obs_dim, act_dim)
    #                 # k = - np.linalg.solve(Q_uu_reg, Q_u)
    #                 # K = - np.linalg.solve(Q_uu_reg, Q_ux_reg)
    #                 global_delta_J_1[active_env_idx] += (k_expand_T @ Q_u_expand)[:, 0, 0]  # (num_active_envs,)
    #                 global_delta_J_2[active_env_idx] += (k_expand_T @ Q_uu @ k_expand)[:, 0, 0]  # (num_active_envs,)
    #
    #                 # prepare for next i
    #                 # V_prime_x = Q_x + Q_u @ feedback_gain
    #                 # V_prime_xx = Q_xx + Q_ux.T @ feedback_gain
    #                 Q_ux_T = np.transpose(Q_ux, axes=[0, 2, 1])
    #                 V_prime_x = Q_x + (K_T @ Q_uu @ k_expand + K_T @ Q_u_expand + Q_ux_T @ k_expand)[:, :, 0]  # (num_envs, obs_dim)
    #                 V_prime_xx = Q_xx + K_T @ Q_uu @ K + K_T @ Q_ux + Q_ux_T @ K
    #                 V_prime_xx = (V_prime_xx + np.transpose(V_prime_xx, axes=[0, 2, 1])) * 0.5
    #
    #             # self._decrease_mu()
    #             backward_accept = True
    #
    #         except np.linalg.LinAlgError: # encountered non-PD Q_uu, increase mu, start backward pass again
    #             logger.log(f'i = {i}, mu = {self.mu}, Q_uu min eigen value = {np.min(np.linalg.eigvals(Q_uu))}')
    #             self._increase_mu()
    #             backward_pass_counter += 1
    #
    #     if not backward_accept:
    #         logger.log(f'backward not accepted with mu = {self.mu}')
    #         return None, backward_accept, forward_accept, None, None
    #
    #     """
    #     Forward Pass (stop if 0 < c_1 < z)
    #     """
    #     alpha = 1.0
    #     forward_pass_counter = 0
    #     while not forward_accept and forward_pass_counter < self.max_forward_iters:
    #         # reset
    #         assert np.allclose(obs, x_array[0])
    #         x = obs  #x_array[0]
    #         opt_x_array, opt_u_array = [], []
    #         reward_array = []
    #         opt_J_val = 0
    #
    #         # forward pass
    #         for i in range(self.horizon):
    #             # (num_envs, act_dim) + (num_envs, act_dim) + ((num_envs, act_dim, obs_dim) @ (num_envs, obs_dim, 1))[:, :, 0]
    #             u = u_array[i] + alpha * open_k_array[i] + (closed_K_array[i] @ np.expand_dims(x - x_array[i], axis=2))[:, :, 0]
    #             u = np.clip(u, self.act_low, self.act_high)
    #
    #             # store updated state/action
    #             opt_x_array.append(x)
    #             opt_u_array.append(u)
    #
    #             x_prime = self.dynamics_model.predict(x, u)
    #             reward = self._env.reward(x, u, x_prime)
    #             reward_array.append(reward)
    #             opt_J_val += -reward
    #             x = x_prime
    #
    #         # Stop if convergence (J_val > opt_J_val and |J_val - opt_J_val| / |J_val| < threshold)
    #         # Maybe decreasing max_forward_iters has same effect
    #         delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
    #         if J_val > opt_J_val and J_val - opt_J_val > self.c_1 * (- delta_J_alpha):
    #             # store updated x, u array (CLIPPED), J_val
    #             optimized_action = opt_u_array[0]
    #             # opt_x_array, opt_u_array = np.stack(opt_x_array, axis=0), np.stack(opt_u_array, axis=0)
    #             # # opt_u_array = np.clip(opt_u_array, self.act_low, self.act_high)
    #             # self.x_array, self.u_array = opt_x_array, opt_u_array
    #             # self.J_val = opt_J_val
    #             self.u_array = np.stack(opt_u_array, axis=0)
    #             forward_accept = True
    #         else:
    #             # continue line search
    #             alpha /= self.alpha_decay_factor
    #             forward_pass_counter += 1
    #
    #         if J_val > opt_J_val:
    #             logger.log(f'at itr {forward_pass_counter}, actual = {J_val - opt_J_val}, exp = {-delta_J_alpha}')
    #
    #     if forward_accept:
    #         logger.log(f'forward pass accepted after {forward_pass_counter} failed iterations')
    #         self._decrease_mu()
    #         return optimized_action, backward_accept, forward_accept, (-J_val, -opt_J_val, -delta_J_alpha), reward_array
    #     else:
    #         logger.log(f'foward pass not accepted')
    #         self._increase_mu()
    #         return optimized_action, backward_accept, forward_accept, None, None

    def get_actions(self, obs):
        self._reset_mu()

        for itr in range(self.num_ilqr_iters):
            """
            Derivatives
            """
            # compute: x_array, J_val, f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux
            feed_dict = {self.u_array_ph: self.u_array_val, self.obs_ph: obs}
            sess = tf.get_default_session()
            utils_by_env = sess.run(self.utils_sym_by_env, feed_dict=feed_dict)

            for env_idx, utils in enumerate(utils_by_env):
                outputs = self.get_action_per_env(env_idx, utils)

                if self.verbose and outputs is not None:
                    old_returns, new_returns, diff = outputs
                    logger.logkv(f'{env_idx}-PlannerPrevReturn', old_returns)
                    logger.logkv(f'{env_idx}-PlannerReturn', new_returns)
                    logger.logkv(f'{env_idx}-ExpectedDiff', diff)
                    logger.logkv(f'{env_idx}-ActualDiff', new_returns - old_returns)

            if self.verbose:
                logger.logkv('Itr', itr)
                logger.dumpkvs()

        optimized_action = self.u_array_val[0, :, :]

        # shift
        self.shift_u_array(u_new=None)

        return optimized_action, []

    def get_action_per_env(self, idx, utils_list):
        x_array, J_val, f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux = utils_list
        u_array = self.u_array_val[:, idx, :]

        """
        Backward Pass
        """
        backward_accept = False
        backward_pass_counter = 0
        while not backward_accept and backward_pass_counter < self.max_backward_iters:
            # initialize
            V_prime_xx, V_prime_x = np.zeros((self.obs_dim, self.obs_dim)), np.zeros(self.obs_dim,)
            open_k_array, closed_K_array = [], []
            delta_J_1, delta_J_2 = 0, 0
            mu, _ = self.param_array[idx]

            try:
                # backward pass
                for i in range(self.horizon-1, -1, -1):
                    # compute Q
                    Q_uu = l_uu[i] + f_u[i].T @ V_prime_xx @ f_u[i]
                    if self.reg_str == 'Q':
                        Q_uu_reg = Q_uu + mu * np.eye(self.act_dim)
                    elif self.reg_str == 'V':
                        Q_uu_reg = Q_uu + mu * f_u[i].T @ f_u[i]
                    else:
                        raise NotImplementedError

                    if not np.allclose(Q_uu, Q_uu.T):
                        print(Q_uu)
                        raise RuntimeError

                    Q_uu_reg_inv = chol_inv(Q_uu_reg)  # except error here

                    Q_x = l_x[i] + f_x[i].T @ V_prime_x
                    Q_u = l_u[i] + f_u[i].T @ V_prime_x
                    Q_xx = l_xx[i] + f_x[i].T @ V_prime_xx @ f_x[i]
                    Q_ux = l_ux[i] + f_u[i].T @ V_prime_xx @ f_x[i]
                    if self.reg_str == 'Q':
                        Q_ux_reg = Q_ux
                    elif self.reg_str == 'V':
                        Q_ux_reg = Q_ux + mu * f_u[i].T @ f_x[i]
                    else:
                        raise NotImplementedError

                    # compute control matrices
                    k = - Q_uu_reg_inv @ Q_u
                    K = - Q_uu_reg_inv @ Q_ux_reg
                    open_k_array.append(k)
                    closed_K_array.append(K)
                    delta_J_1 += k.T @ Q_u
                    delta_J_2 += k.T @ Q_uu @ k

                    # prepare for next i
                    V_prime_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
                    V_prime_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
                    V_prime_xx = (V_prime_xx + V_prime_xx.T) * 0.5

                backward_accept = True

            except np.linalg.LinAlgError: # encountered non-PD Q_uu, increase mu, start backward pass again
                logger.log(f'backward_pass_counter = {backward_pass_counter}, i = {i}, mu = {mu}, Q_uu min eigen value = {np.min(np.linalg.eigvals(Q_uu))}')
                self._increase_mu(idx)
                backward_pass_counter += 1

        if not backward_accept:
            logger.log(f'backward not accepted with mu = {mu}')
            return None

        """
        Forward Pass (stop if 0 < c_1 < z)
        """
        forward_accept = False
        alpha = 1.0
        forward_pass_counter = 0
        while not forward_accept and forward_pass_counter < self.max_forward_iters:
            # reset
            x = x_array[0]
            opt_x_array, opt_u_array = [], []
            opt_J_val = 0

            # forward pass
            for i in range(self.horizon):
                u = u_array[i] + alpha * open_k_array[i] + closed_K_array[i] @ (x - x_array[i])
                u = np.clip(u, self.act_low, self.act_high)

                # store updated state/action
                opt_x_array.append(x)
                opt_u_array.append(u)
                x_prime= self.dynamics_model.predict(x[None], u[None])[0]
                reward = self._env.reward(x, u, x_prime)
                opt_J_val += -self.discount**i * reward
                x = x_prime

            delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
            if J_val > opt_J_val and J_val - opt_J_val > self.c_1 * (- delta_J_alpha):
                # store updated u array (CLIPPED)
                self.u_array_val[:, idx, :] = np.stack(opt_u_array, axis=0)
                forward_accept = True
            else:
                # continue line search
                alpha /= self.alpha_decay_factor
                forward_pass_counter += 1

            # if J_val > opt_J_val:
            #     logger.log(f'at itr {forward_pass_counter}, actual = {J_val - opt_J_val}, exp = {-delta_J_alpha}')

        if forward_accept:
            logger.log(f'backward_pass, forward pass accepted after {backward_pass_counter, forward_pass_counter} failed iterations')
            self._decrease_mu(idx)
            return -J_val, -opt_J_val, -delta_J_alpha
        else:
            logger.log(f'forward pass not accepted with mu = {mu}')
            self._increase_mu(idx)
            return None

    def _decrease_mu(self, idx):
        mu, delta = self.param_array[idx]
        delta = min(1, delta) / self.delta_0
        mu *= delta
        if mu < self.mu_min:
            mu = 0.0
        self.param_array[idx] = (mu, delta)

    def _increase_mu(self, idx):
        mu, delta = self.param_array[idx]
        delta = max(1.0, delta) * self.delta_0
        mu = max(self.mu_min, mu * delta)
        if mu > self.mu_max:
            RuntimeWarning(f'self.mu = {mu} > self.mu_max')
        self.param_array[idx] = (mu, delta)

    def _reset_mu(self):
        self.param_array = [(self.mu_init, self.delta_init) for _ in range(self.num_envs)]

    def reset_u_array(self):
        # self._reset_mu()

        u_array = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.horizon, self.num_envs, self.act_dim))
        self.u_array_val = u_array

    def shift_u_array(self, u_new):
        """
        Shifting schedule: rotation; zeros (with/without Gaussian noise); random uniform
        :param u_new: (act_dim,)
        :return:
        """
        # self._reset_mu()
        if u_new is None:
            u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.num_envs,self.act_dim))

        self.u_array_val = np.concatenate([self.u_array_val[1:, :, :], u_new[None]])


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
