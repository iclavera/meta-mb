from meta_mb.logger import logger
import numpy as np
from collections import OrderedDict
import time
import tensorflow as tf
import scipy.linalg as sla


def stack_wrapper(arr):
    return tf.stack(arr, axis=0)


class DyniLQRPlanner(object):
    """
    Compute ground truth derivatives.
    """
    def __init__(self, env, dynamics_model, num_envs, horizon, u_array, reg_str='V', discount=1,
                 mu_min=1e-6, mu_max=1e10, mu_init=1e-5, delta_0=2, delta_init=1.0, alpha_decay_factor=3.0,
                 c_1=1e-7, max_forward_iters=10, max_backward_iters=10,
                 forward_stop_cond='rel',
                 use_hessian_f=False, verbose=False):
        self._env = env
        self.dynamics_model = dynamics_model
        self.num_envs = num_envs
        self.horizon = horizon
        self.u_array_val = u_array
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

        self.utils_sym_dict = self._build_deriv_graph()

    def _build_deriv_graph(self):
        """
        u_array = (horizon,
        :return:
        """
        x_array, df_array, dl_array = [], [], []

        obs = self.obs_ph
        returns = 0
        for i in range(self.horizon):
            x_array.append(obs)
            acts = self.u_array_ph[i]
            next_obs = self.dynamics_model.predict_sym(obs, acts)
            df = self._df_sym(obs, acts, next_obs)
            df_array.append(df)
            rewards = self._env.tf_reward(obs, acts, next_obs)
            dl = self._dl_sym(rewards, obs, acts, next_obs)
            dl_array.append(dl)
            returns += self.discount**i * rewards
            obs = next_obs

        x_array = stack_wrapper(x_array)
        J_val = -returns
        f_x, f_u = list(map(stack_wrapper, zip(*df_array)))  # [f_x, f_u]
        l_x, l_u, l_xx, l_uu, l_ux = list(map(stack_wrapper, zip(*dl_array)))

        return OrderedDict(
            x_array=x_array,
            J_val=J_val,
            f_x=f_x,
            f_u=f_u,
            l_x=l_x,
            l_u=l_u,
            l_xx=l_xx,
            l_uu=l_uu,
            l_ux=l_ux,
        )

    def _dl_sym(self, rewards, obs, acts, next_obs):
        """

        :param rewards: (num_envs,)
        :param obs:
        :param acts:
        :param next_obs:
        :return:
        """
        # obs = tf.stop_gradient(obs)
        # acts = tf.stop_gradient(acts)

        # l_x = (num_envs, obs_dim)
        l_x, = tf.gradients(rewards, xs=obs)
        # l_u = (num_envs, act_dim)
        l_u, = tf.gradients(rewards, xs=acts)
        # l_xx = (num_envs, obs_dim, obs_dim)
        l_xx, = tf.hessians(rewards, xs=obs)
        # l_uu = (num_envs, act_dim, act_dim)
        l_uu, = tf.hessians(rewards, xs=acts)
        # l_ux = (num_envs, act_dim, obs_dim)
        l_ux = self._gradients_wrapper(l_u, obs, self.act_dim)

        return l_x, l_u, l_xx, l_uu, l_ux

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
            print(tf.gradients(ys=y[:, i], xs=[x]))
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
        jac_f_x_array = self._gradients_wrapper(next_obs, obs, self.obs_dim)
        # jac_f_x = []
        # for n in range(self.num_envs):
        #     jac_f_x_per_env = []
        #     for i in range(self.obs_dim):
        #         jac_f_i_x_per_env = tf.gradients(next_obs[n, i], var_list=obs[n])  # i-th row of jac_f_x
        #         jac_f_x_per_env.append(jac_f_i_x_per_env)
        #     jac_f_x_per_env = tf.stack(jac_f_x_per_env, axis=0)  # (obs_dim, obs_dim)
        #     jac_f_x.append(jac_f_x_per_env)
        # jac_f_x = tf.stack(jac_f_x, axis=0)

        # jac_f_u: (num_envs, obs_dim, act_dim)
        jac_f_u_array = self._gradients_wrapper(next_obs, acts, self.obs_dim)
        # jac_f_u = []
        # for n in range(self.num_envs):
        #     jac_f_u_per_env = []
        #     for i in range(self.obs_dim):
        #         jac_f_i_u_per_env = tf.gradients(next_obs[n, i], var_list=acts[n])  # i-th row of jac_f_u
        #         jac_f_u_per_env.append(jac_f_i_u_per_env)
        #     jac_f_u_per_env = tf.stack(jac_f_u_per_env, axis=0)
        #     jac_f_u.append(jac_f_u_per_env)
        # jac_f_u = tf.stack(jac_f_u, axis=0)

        return jac_f_x_array, jac_f_u_array

    def update_x_u_for_one_step(self, obs):
        u_array = self.u_array_val
        optimized_action = u_array[0]
        backward_accept, forward_accept = False, False

        """
        Derivatives
        """
        feed_dict = {self.u_array_ph: self.u_array_val, self.obs_ph: obs}
        sess = tf.get_default_session()
        x_array, J_val, f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux = sess.run(
            list(self.utils_sym_dict.values()),
            feed_dict=feed_dict,
        )
        """
        Backward Pass
        """
        backward_pass_counter = 0
        while not backward_accept and backward_pass_counter < self.max_backward_iters:# and self.mu <= self.mu_max:
            # initialize
            V_prime_xx, V_prime_x = np.zeros((self.obs_dim, self.obs_dim)), np.zeros(self.obs_dim,)  # l_x[-1], l_xx[-1]
            open_k_array, closed_K_array = [], []
            delta_J_1, delta_J_2 = 0, 0

            try:
                # backward pass
                for i in range(self.horizon-1, -1, -1):
                    # compute Q
                    Q_uu = l_uu[i] + f_u[i].T @ V_prime_xx @ f_u[i]
                    if self.reg_str == 'Q':
                        Q_uu_reg = Q_uu + self.mu * np.eye(self.act_dim)
                    elif self.reg_str == 'V':
                        Q_uu_reg = Q_uu + self.mu * f_u[i].T @ f_u[i]
                    else:
                        raise NotImplementedError

                        # Q_uu_no_reg = l_uu[i] + f_u[i].T @ V_prime_xx @ f_u[i]
                        # logger.log('Q_uu_no_reg min eigen value', np.min(np.linalg.eigvals(Q_uu_no_reg)))
                        # logger.log('Q_uu min eigen value', np.min(np.linalg.eigvals(Q_uu)))

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
                        Q_ux_reg = Q_ux + self.mu * f_u[i].T @ f_x[i]
                    else:
                        raise NotImplementedError

                    # compute control matrices
                    # Q_uu_inv = np.linalg.inv(Q_uu)
                    k = - Q_uu_reg_inv @ Q_u  # k
                    K = - Q_uu_reg_inv @ Q_ux_reg  # K
                    # k = - np.linalg.solve(Q_uu_reg, Q_u)
                    # K = - np.linalg.solve(Q_uu_reg, Q_ux_reg)
                    open_k_array.append(k)
                    closed_K_array.append(K)
                    delta_J_1 += k.T @ Q_u
                    delta_J_2 += k.T @ Q_uu @ k

                    # prepare for next i
                    # V_prime_x = Q_x + Q_u @ feedback_gain
                    # V_prime_xx = Q_xx + Q_ux.T @ feedback_gain
                    V_prime_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
                    V_prime_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
                    V_prime_xx = (V_prime_xx + V_prime_xx.T) * 0.5

                # self._decrease_mu()
                backward_accept = True

            except np.linalg.LinAlgError: # encountered non-PD Q_uu, increase mu, start backward pass again
                logger.log(f'i = {i}, mu = {self.mu}, Q_uu min eigen value = {np.min(np.linalg.eigvals(Q_uu))}')
                self._increase_mu()
                backward_pass_counter += 1

        if not backward_accept:
            logger.log(f'backward not accepted with mu = {self.mu}')
            return None, backward_accept, forward_accept, None, None

        """
        Forward Pass (stop if 0 < c_1 < z)
        """
        alpha = 1.0
        forward_pass_counter = 0
        while not forward_accept and forward_pass_counter < self.max_forward_iters:
            # reset
            assert np.allclose(obs, x_array[0])
            x = obs  #x_array[0]
            opt_x_array, opt_u_array = [], []
            reward_array = []
            opt_J_val = 0

            # forward pass
            for i in range(self.horizon):
                u = u_array[i] + alpha * open_k_array[i] + closed_K_array[i] @ (x - x_array[i])
                u = np.clip(u, self.act_low, self.act_high)

                # store updated state/action
                opt_x_array.append(x)
                opt_u_array.append(u)
                time.sleep(0.004)
                x, reward = self._f(x, u)
                reward_array.append(reward)
                opt_J_val += -reward

            # Stop if convergence (J_val > opt_J_val and |J_val - opt_J_val| / |J_val| < threshold)
            # Maybe decreasing max_forward_iters has same effect
            delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
            if J_val > opt_J_val and J_val - opt_J_val > self.c_1 * (- delta_J_alpha):
                # store updated x, u array (CLIPPED), J_val
                optimized_action = opt_u_array[0]
                # opt_x_array, opt_u_array = np.stack(opt_x_array, axis=0), np.stack(opt_u_array, axis=0)
                # # opt_u_array = np.clip(opt_u_array, self.act_low, self.act_high)
                # self.x_array, self.u_array = opt_x_array, opt_u_array
                # self.J_val = opt_J_val
                self.u_array = np.stack(opt_u_array, axis=0)
                forward_accept = True
            else:
                # continue line search
                alpha /= self.alpha_decay_factor
                forward_pass_counter += 1

            if J_val > opt_J_val:
                logger.log(f'at itr {forward_pass_counter}, actual = {J_val - opt_J_val}, exp = {-delta_J_alpha}')

        if forward_accept:
            logger.log(f'forward pass accepted after {forward_pass_counter} failed iterations')
            self._decrease_mu()
            return optimized_action, backward_accept, forward_accept, (-J_val, -opt_J_val, -delta_J_alpha), reward_array
        else:
            logger.log(f'foward pass not accepted')
            self._increase_mu()
            return optimized_action, backward_accept, forward_accept, None, None

    def _decrease_mu(self):
        self.delta = min(1, self.delta) / self.delta_0
        self.mu *= self.delta
        if self.mu < self.mu_min:
            self.mu = 0.0

    def _increase_mu(self):
        # adapt delta, mu
        self.delta = max(1.0, self.delta) * self.delta_0
        self.mu = max(self.mu_min, self.mu * self.delta)
        if self.mu > self.mu_max:
            RuntimeWarning(f'self.mu = {self.mu} > self.mu_max')

    def _reset_mu(self):
        self.mu = self.mu_init
        self.delta = self.delta_init

    def perturb_u_array(self):
        self._reset_mu()

        u_array = self.u_array + np.random.normal(loc=0, scale=0.1, size=self.u_array.shape)
        u_array = np.clip(u_array, a_min=self.act_low, a_max=self.act_high)
        self.u_array = u_array
        # self.x_array, self.J_val = None, None

    def reset_u_array(self):
        self._reset_mu()

        u_array = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.horizon, self.act_dim))
        self.u_array_val = u_array

    def shift_u_array(self, u_new):
        """
        Shifting schedule: rotation; zeros (with/without Gaussian noise); random uniform
        :param u_new: (act_dim,)
        :return:
        """
        self._reset_mu()
        if u_new is None:
            # u_new = np.mean(self.u_array, axis=0) + np.random.normal(loc=0, scale=0.05, size=(self.act_dim,))
            # self.u_array = np.roll(self.u_array, -1, axis=0)
            u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.act_dim,))

        self.u_array_val = np.concatenate([self.u_array_val[1:, :], u_new[None]])
        # self.x_array, self.J_val = None, None


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
