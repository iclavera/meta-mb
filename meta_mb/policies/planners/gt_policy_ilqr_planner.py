import numpy as np
from meta_mb.logger import logger
import time
import scipy.linalg as sla
import copy


class GTPolicyiLQRPlanner(object):
    def __init__(self, env, policy, n_parallel, horizon, eps, u_array, reg_str='V',
                 mu_min=1e-6, mu_max=1e10, mu_init=1e-5, delta_0=2, delta_init=1.0, alpha_decay_factor=3.0,
                 c_1=1e-7, max_forward_iters=10, max_backward_iters=10,
                 use_hessian_f=False, verbose=False):
        self._env = copy.deepcopy(env)
        self.policy = policy
        self.n_parallel = n_parallel
        self.horizon = horizon
        self.eps = eps
        self.u_array = u_array
        self.reg_str = reg_str
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_init = mu_init
        self.alpha_decay_factor = alpha_decay_factor
        self.delta_0 = delta_0
        self.delta_init = delta_init
        self.c_1 = c_1
        self.max_forward_iters = max_forward_iters
        self.max_backward_iters = max_backward_iters
        self.use_hessian_f = use_hessian_f

        self.theta_dim = policy.flatten_dim
        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        self._reset_mu()

    def update_x_u_for_one_step(self, obs):
        policy = self.policy
        u_array = self.u_array
        u_array[0], _ = self.policy.get_action(obs)
        optimized_action = u_array[0]
        x_array, J_val = self._run_open_loop(u_array=u_array, init_obs=obs)
            # x_array, J_val = self.x_array, self.J_val
        backward_accept, forward_accept = False, False

        assert x_array.shape == (self.horizon, self.obs_dim)
        assert u_array.shape == (self.horizon, self.act_dim)

        """
        Derivatives
        """
        # derivatives for i > 0
        dl, df = self._compute_deriv(x_array=x_array, u_array=u_array)  # dl, df has length = horizon
        l_x, l_u, l_xx, l_uu, l_ux = dl
        if self.use_hessian_f:
            f_x, f_u, f_xx, f_uu, f_ux = df
        else:
            f_x, f_u = df
            f_xx, f_uu, f_ux = None, None, None

        # derivatives for i = 0
        # du
        # u_x, u_theta, u_xx, u_theta_theta, u_theta_x = policy.compute_du(x_array[0])
        _, u_theta, _, _, _ = policy.compute_du(x_array[0])
        # dl
        # l_bar_x = l_x[0] + u_x.T @ l_u[0]
        # l_bar_theta = u_theta.T @ l_u[0]  # (theta_dim,)
        # l_bar_xx = l_xx[0] + u_x.T @ l_ux[0] @ u_x + np.tensordot(l_u[0], u_xx)
        # l_bar_theta_theta = u_theta.T @ l_uu[0] @ u_theta # + np.tensordot(l_u[0], u_theta_theta)  # FIXME: drop second order?
        # l_bar_theta_x = u_theta.T @ l_uu[0] @ u_x + np.tensordot(l_u[0], u_theta_x)
        # df
        # f_bar_x = f_x[0] + f_u[0] @ u_x
        # f_bar_theta = f_u[0] @ u_theta

        """
        Backward Pass
        """
        backward_pass_counter = 0
        while not backward_accept and backward_pass_counter < self.max_backward_iters:# and self.mu <= self.mu_max:
            # initialize
            V_prime_xx, V_prime_x = np.zeros((self.obs_dim, self.obs_dim)), np.zeros(self.obs_dim,)
            open_k_array, closed_K_array = [None] * self.horizon, [None] * self.horizon
            delta_J_1, delta_J_2 = 0, 0

            try:
                # backward pass
                for i in range(self.horizon-1, 0, -1):
                    # compute Q
                    if self.use_hessian_f:
                        Q_uu = l_uu[i] + f_u[i].T @ V_prime_xx @ f_u[i] + np.tensordot(V_prime_x, f_uu[i], axes=1)
                    else:
                        Q_uu = l_uu[i] + f_u[i].T @ V_prime_xx @ f_u[i]
                    if self.reg_str == 'Q':
                        Q_uu_reg = Q_uu + self.mu * np.eye(self.act_dim)
                    elif self.reg_str == 'V':
                        Q_uu_reg = Q_uu + self.mu * f_u[i].T @ f_u[i]
                    else:
                        raise ValueError

                    if not np.allclose(Q_uu, Q_uu.T):
                        print(Q_uu)
                        raise RuntimeError

                    Q_uu_reg_inv = chol_inv(Q_uu_reg)  # except error here

                    Q_x = l_x[i] + f_x[i].T @ V_prime_x
                    Q_u = l_u[i] + f_u[i].T @ V_prime_x
                    if self.use_hessian_f:
                        Q_xx = l_xx[i] + f_x[i].T @ V_prime_xx @ f_x[i] + np.tensordot(V_prime_x, f_xx[i], axes=1)
                        Q_ux = l_ux[i] + f_u[i].T @ V_prime_xx @ f_x[i] + np.tensordot(V_prime_x, f_ux[i], axes=1)
                    else:
                        Q_xx = l_xx[i] + f_x[i].T @ V_prime_xx @ f_x[i]
                        Q_ux = l_ux[i] + f_u[i].T @ V_prime_xx @ f_x[i]
                    if self.reg_str == 'Q':
                        Q_ux_reg = Q_ux
                    elif self.reg_str == 'V':
                        Q_ux_reg = Q_ux + self.mu * f_u[i].T @ f_x[i]
                    else:
                        raise ValueError

                    # compute control matrices
                    if i > 0:
                        k = - Q_uu_reg_inv @ Q_u
                        K = - Q_uu_reg_inv @ Q_ux_reg
                        open_k_array[i] = k
                        closed_K_array[i] = K
                        delta_J_1 += k.T @ Q_u
                        delta_J_2 += k.T @ Q_uu @ k

                        # prepare for next i
                        # V_prime_x = Q_x + Q_u @ feedback_gain
                        # V_prime_xx = Q_xx + Q_ux.T @ feedback_gain
                        V_prime_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
                        V_prime_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
                        V_prime_xx = (V_prime_xx + V_prime_xx.T) * 0.5

                i = 0
                Q_theta_theta = u_theta.T @ Q_uu_reg @ u_theta + self.mu * np.eye(self.theta_dim) # + np.tensordot(Q_u, u_thetat_theta)A  # FIXME: Q_uu_reg or Q_uu?
                Q_theta_theta_inv = chol_inv(Q_theta_theta)
                Q_theta = Q_u @ u_theta

                k = -Q_theta_theta_inv @ Q_theta
                open_k_array[0] = k
                delta_J_1 += k.T @ Q_theta
                delta_J_2 += k.T @ Q_theta_theta @ k

                # # i = 0, consider Q_bar
                # i = 0
                # if self.use_hessian_f:
                #     raise NotImplementedError
                # else:
                #     Q_bar_theta_theta = l_bar_theta_theta + f_bar_theta.T @ V_prime_xx @ f_bar_theta
                # if self.reg_str == 'Q':
                #     Q_bar_theta_theta_reg = Q_bar_theta_theta + self.mu * np.eye(self.theta_dim)
                # elif self.reg_str == 'V':
                #     Q_bar_theta_theta_reg = Q_bar_theta_theta + self.mu * f_bar_theta.T @ f_bar_theta
                # else:
                #     raise ValueError
                #
                # assert np.allclose(Q_bar_theta_theta, Q_bar_theta_theta.T)
                #
                # Q_bar_theta_theta_reg_inv = chol_inv(Q_bar_theta_theta_reg)
                #
                # # Q_bar_x = l_bar_x + f_bar_x.T @ V_prime_x
                # Q_bar_theta = l_bar_theta + f_bar_theta.T @ V_prime_x
                # # if self.use_hessian_f:
                # #     raise NotImplementedError
                # # else:
                # #     Q_bar_xx = l_bar_xx + f_bar_x.T @ V_prime_xx @ f_bar_x
                # #     Q_bar_theta_x = l_bar_theta_x + f_bar_theta.T @ V_prime_xx @ f_bar_x
                # # if self.reg_str == 'Q':
                # #     Q_bar_theta_x_reg = Q_bar_theta_x
                # # elif self.reg_str == 'V':
                # #     Q_bar_theta_x_reg = Q_bar_theta_x + self.mu * f_bar_theta.T @ f_bar_x
                # # else:
                # #     raise ValueError
                #
                # k = - Q_bar_theta_theta_reg_inv @ Q_bar_theta
                # # K = - Q_bar_theta_theta_reg_inv @ Q_bar_theta_x_reg
                # open_k_array[0] = k
                # # closed_K_array[0] = K
                # delta_J_1 += k.T @ Q_bar_theta
                # delta_J_2 += k.T @ Q_bar_theta_theta @ k

                backward_accept = True

            except np.linalg.LinAlgError: # encountered non-PD Q_uu, increase mu, start backward pass again
                if i > 0:
                    logger.log(f'i = {i}, mu = {self.mu}, Q_uu min eigen value = {np.min(np.linalg.eigvals(Q_uu))}')
                else:
                    logger.log(f'i = {i}, mu = {self.mu}, Q_bar_theta_theta min eigen value = {np.min(np.linalg.eigvals(Q_theta_theta))}')
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
        theta = policy.get_param_values_flatten()
        while not forward_accept and forward_pass_counter < self.max_forward_iters:
            # reset
            assert np.allclose(obs, x_array[0])
            x = obs  #x_array[0]
            opt_x_array, opt_u_array = [], []
            reward_array = []
            opt_J_val = 0

            # forward pass
            for i in range(self.horizon):
                if i == 0:
                    # theta <- theta + alpha * k
                    policy.set_param_values_flatten(theta + alpha * open_k_array[0])  # since x - x_array[0] = 0, closed loop term is dropped
                    u, _ = policy.get_action(x)  # FIXME: either clip or use tanh
                else:
                    u = u_array[i] + alpha * open_k_array[i] + closed_K_array[i] @ (x - x_array[i])
                u = np.clip(u, self.act_low, self.act_high)

                # store updated state/action
                opt_x_array.append(x)
                opt_u_array.append(u)
                time.sleep(0.004)
                x, reward = self._f(x, u, return_reward=True)
                reward_array.append(reward)
                opt_J_val += -reward

            delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
            if J_val > opt_J_val and J_val - opt_J_val > self.c_1 * (- delta_J_alpha):
                # store updated x, u array (CLIPPED), J_val
                optimized_action = opt_u_array[0]
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
            policy.set_param_values_flatten(theta)
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
            raise RuntimeWarning(f'self.mu = {self.mu} > self.mu_max')

    def _reset_mu(self):
        self.mu = self.mu_init
        self.delta = self.delta_init

    def reset_u_array(self, u_array):
        self._reset_mu()
        self.u_array = u_array

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

        self.u_array = np.concatenate([self.u_array[1:, :], u_new[None]])

    def _run_open_loop(self, u_array, init_obs):
        x_array, sum_rewards = [], 0
        x = self._env.reset_from_obs(init_obs)
        # try:
        #     assert np.allclose(x, init_obs)  # FAILS IN REACHER
        # except AssertionError:
        #     logger.log('ERROR from reset_from_obs')

        for i in range(self.horizon):
            x_array.append(x)
            x, reward, _, _ = self._env.step(u_array[i])
            sum_rewards += reward
        x_array = np.stack(x_array, axis=0)

        return x_array, -sum_rewards

    def _f(self, x, u, return_reward=False):
        _ = self._env.reset_from_obs(x)
        x_prime, reward, _, _ = self._env.step(u)
        if return_reward:
            return x_prime, reward
        return x_prime

    def _df_by_pair(self, x, u, centered=True):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        _f = self._f
        eps = self.eps

        # jac_f_x: (obs_dim, obs_dim)
        jac = np.zeros((obs_dim, obs_dim))
        e_i = np.zeros((obs_dim,))
        if centered:
            for i in range(obs_dim):
                e_i[i] = eps
                jac[:, i] = (_f(x+e_i, u) - _f(x-e_i, u)) / (2*eps)
                e_i[i] = 0
        else:
            f_val = _f(x, u)
            for i in range(obs_dim):
                e_i[i] = eps
                jac[:, i] = (_f(x+e_i, u) - f_val) / eps
                e_i[i] = 0
        jac_f_x = jac

        # jac_f_u: (obs_dim, act_dim)
        jac = np.zeros((obs_dim, act_dim))
        e_i = np.zeros((act_dim,))
        if centered:
            for i in range(act_dim):
                e_i[i] = eps
                jac[:, i] = (_f(x, u+e_i) - _f(x, u-e_i)) / (2*eps)
                e_i[i] = 0
        else:
            f_val = _f(x, u)
            for i in range(act_dim):
                e_i[i] = eps
                jac[:, i] = (_f(x, u+e_i) - f_val) / eps
                e_i[i] = 0
        jac_f_u = jac

        if not self.use_hessian_f:
            return jac_f_x, jac_f_u

        # hessian_f_xx
        hess = np.zeros((obs_dim, obs_dim, obs_dim))
        f_val = _f(x, u)
        e_i, e_j = np.zeros((obs_dim,)), np.zeros((obs_dim,))
        for i in range(obs_dim):
            e_i[i] = eps
            f_val_fix_i = _f(x+e_i, u) - f_val
            for j in range(obs_dim):
                e_j[j] = eps
                hess[:, i, j] = (_f(x+e_i+e_j, u) - _f(x+e_j, u) - f_val_fix_i) / eps**2
                e_j[j] = 0
            e_i[i] = 0
        hessian_f_xx = (hess + np.transpose(hess, axes=[0, 2, 1])) * 0.5

        # hessian_f_uu
        hess = np.zeros((obs_dim, act_dim, act_dim))
        f_val = _f(x, u)
        e_i, e_j = np.zeros((act_dim,)), np.zeros((act_dim,))
        for i in range(act_dim):
            e_i[i] = eps
            f_val_fix_i = _f(x, u+e_i) - f_val
            for j in range(act_dim):
                e_j[j] = eps
                hess[:, i, j] = (_f(x, u+e_i+e_j) - _f(x, u+e_j) - f_val_fix_i) / eps**2
                e_j[j] = 0
            e_i[i] = 0
        hessian_f_uu = (hess + np.transpose(hess, axes=[0, 2, 1])) * 0.5

        # hessian_f_ux
        hess = np.zeros((obs_dim, act_dim, obs_dim))
        f_val = _f(x, u)
        e_i, e_j = np.zeros((act_dim,)), np.zeros((obs_dim,))
        for i in range(act_dim):
            e_i[i] = eps
            f_val_fix_i = _f(x, u+e_i) - f_val
            for j in range(obs_dim):
                e_j[j] = eps
                hess[:, i, j] = (_f(x+e_j, u+e_i) - _f(x, u+e_i) - f_val_fix_i) / eps**2
                e_j[j] = 0
            e_i[i] = 0
        hessian_f_ux = hess

        return jac_f_x, jac_f_u, hessian_f_xx, hessian_f_uu, hessian_f_ux

    def _compute_deriv(self, x_array, u_array):
        df_by_pair = [self._df_by_pair(x_array[i], u_array[i]) for i in range(self.horizon)]
        df = list(zip(*df_by_pair))  # df = [jac_f_x, jac_f_u, ...], jac_f_x is a list of length horizon
        dl = self._env.dl_dict(dict(obs=x_array, act=u_array)).values()
        return dl, df


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