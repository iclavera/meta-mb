from meta_mb.logger import logger
import scipy.linalg as sla
import numpy as np
from collections import OrderedDict
import tensorflow as tf
import time


class DyniLQRPlanner(object):
    def __init__(self, env, dynamics_model, num_envs, horizon, u_array, num_ilqr_iters ,reg_str='V', discount=1,
                 mu_min=1e-6, mu_max=1e10, mu_init=1e-5, delta_0=2, delta_init=1.0, alpha_decay_factor=3.0,
                 c_1=1e-7, max_forward_iters=10, max_backward_iters=10,
                 verbose=False):
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
        self.verbose = verbose
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        self.param_array = None
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

        build_time = time.time()
        self.utils_sym_by_env = self._build_deriv_graph()
        logger.log('TimeBuildGraph', build_time - time.time())

    def _build_deriv_graph(self):
        x_array, df_array, dl_array = [], [], []

        obs = self.obs_ph
        returns = tf.zeros((self.num_envs,))
        for i in range(self.horizon):
            acts = self.u_array_ph[i, :, :]
            next_obs = self.dynamics_model.predict_sym(obs, acts, pred_type='mean', deterministic=True)

            x_array.append(obs)
            df = self._df_sym(obs, acts, next_obs).values()
            df_array.append(df)
            dl = self._env.tf_dl_dict(obs, acts, next_obs, self.num_envs).values()
            dl_array.append(dl)
            rewards = self._env.tf_reward(obs, acts, next_obs)
            returns += self.discount**i * rewards

            obs = next_obs

        # post process
        x_array = tf.stack(x_array, axis=0)
        J_val = -returns
        # for each element of array, stack along horizon: (horizon, num_envs, *)
        f_array = list(map(lambda arr: tf.stack(arr, axis=0), zip(*df_array)))
        l_array = list(map(lambda arr: tf.stack(arr, axis=0), zip(*dl_array)))
        # store by env
        derivs_by_env = list(map(lambda arr: [arr[:, env_idx] for env_idx in range(self.num_envs)], f_array + l_array))
        x_array_by_env = [x_array[:, env_idx] for env_idx in range(self.num_envs)]
        J_val_by_env = [J_val[env_idx] for env_idx in range(self.num_envs)]
        utils_sym_by_env = list(zip(x_array_by_env, J_val_by_env, *derivs_by_env))

        return utils_sym_by_env

    def _gradients_wrapper(self, y, x, dim_y, dim_x, stop_gradients=None):
        """

        :param y: (num_envs, dim_y)
        :param x: (num_envs, dim_x)
        :param dim_y:
        :param dim_x:
        :return: (num_envs, dim_y, dim_x)
        """
        # jac_array = []
        # mask = np.zeros((self.num_envs, dim_x))
        # for j in range(dim_y):
        #     jac_per_dim = tf.zeros((self.num_envs, dim_x))
        #     for i in range(self.num_envs):
        #         jac, = tf.gradients(ys=y[i, j], xs=[x], stop_gradients=stop_gradients)
        #         mask[i, :] = 1
        #         jac_per_dim += jac * tf.constant(mask, dtype=tf.float32)  # only take ith row (ith env)
        #         mask[i, :] = 0
        #     jac_array.append(jac_per_dim)
        # return tf.stack(jac_array, axis=1)
        jac_array = []
        for i in range(dim_y):
            jac, = tf.gradients(ys=y[:, i], xs=[x], stop_gradients=stop_gradients)  # FIXME: stop_gradients?
            jac_array.append(jac)
        return tf.stack(jac_array, axis=1)  # FIXME: is it safe not to separate envs?

    def _df_sym(self, obs, acts, next_obs):
        """

        :param obs: (num_envs, obs_dim)
        :param acts: (num_envs, act_dim)
        :param next_obs: (num_envs, obs_dim)
        """
        # jac_f_x: (num_envs, obs_dim, obs_dim)
        jac_f_x = self._gradients_wrapper(next_obs, obs, self.obs_dim, self.obs_dim, [obs, acts])

        # jac_f_u: (num_envs, obs_dim, act_dim)
        jac_f_u = self._gradients_wrapper(next_obs, acts, self.obs_dim, self.act_dim, [obs, acts])

        return OrderedDict(f_x=jac_f_x, f_u=jac_f_u)

    def get_actions(self, obs):
        self._reset_mu()
        returns_dyn = np.zeros((self.num_envs,))
        # success_counter = 0
        sess = tf.get_default_session()
        feed_dict = {self.u_array_ph: self.u_array_val, self.obs_ph: obs}  # self.u_array_val is updated over iterations

        active_envs = list(range(self.num_envs))
        for itr in range(self.num_ilqr_iters):
            # compute: x_array, J_val, f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux
            utils_by_env = sess.run(self.utils_sym_by_env, feed_dict=feed_dict)
            for env_idx in active_envs.copy():
                try:
                    assert np.allclose(utils_by_env[env_idx][0][0], obs[env_idx])  # compare x_array[0] and obs for the current environment
                    J_val, log_info = self.update_u_per_env(env_idx, utils_by_env[env_idx])
                    if log_info is None:
                        returns_dyn[env_idx] = -J_val
                    else:
                        returns_dyn[env_idx] = -log_info['opt_J_val']
                        # success_counter += 1

                    # logging
                    if env_idx == 0:
                        logger.logkv('iLQRItr', itr)
                        logger.logkv('J_val', J_val)
                        if log_info is not None:
                            logger.logkv('J_val_new', log_info['opt_J_val'])
                            if log_info['u_clipped_pct'] > 0:
                                logger.logkv('u_clipped_pct', log_info['u_clipped_pct'])
                            # logger.logkv(f'RetDyn-{itr}-e{0}', returns_dyn[0])
                            # logger.logkv(f'RetEnv-{itr}-e{0}', self._run_open_loop(self.u_array_val[:, 0, :], obs[0, :]))
                            # logger.logkv(f'{itr}-bw_fw_ctr', (log_info['backward'], log_info['forward']))
                            # logger.logkv(f'{itr}-mu', log_info['mu'])
                            # logger.logkv(f'ActDiff', log_info['J_val'] - log_info['opt_J_val'])
                        logger.dumpkvs()

                        # sanity check  # FIXME: assertion fails with ensemble or stochastic model
                        # tf_opt_J_val = sess.run(self.utils_sym_by_env[env_idx][1], feed_dict={self.u_array_ph: self.u_array_val, self.obs_ph: obs})
                        # logger.log(f'itr = {itr}, idx = {env_idx}, opt_J_val = {-returns_dyn[env_idx]}, tf_opt_J_val = {tf_opt_J_val}')
                        # tf_opt_J_val = sess.run(self.utils_sym_by_env[env_idx][1], feed_dict={self.u_array_ph: self.u_array_val, self.obs_ph: obs})
                        # logger.log(f'itr = {itr}, idx = {env_idx}, opt_J_val = {-returns_dyn[env_idx]}, tf_opt_J_val = {tf_opt_J_val}')
                        # assert tf_opt_J_val == -returns_dyn[env_idx]
                except OverflowError:
                    active_envs.remove(env_idx)
            if len(active_envs) == 0:
                break

            # return -opt_J_val, dict(backward=backward_pass_counter, forward=forward_pass_counter,
            #                         mu=self.param_array[idx][0],
            #                         u_clipped_pct=np.sum(np.abs(opt_u_array) == np.mean(self.act_high))/(self.horizon*self.act_dim),
            #                         J_val=J_val, opt_J_val=opt_J_val, delta_J_alpha=delta_J_alpha)

        # logging
        # sum_returns_diff = 0  # diff between predicted and real returns (sum over horizon)
        # for idx in range(self.num_envs):
        #     returns = self._run_open_loop(self.u_array_val[:, idx, :], obs[idx, :])
        #     # if idx == 0:
        #     #     logger.logkv(f'RetEnv', returns)
        #     #     logger.logkv(f'RetDyn', returns_dyn[0])
        #     sum_returns_diff += np.sum(returns - returns_dyn)
        # # logger.logkv('SuccessRate', success_counter/(self.num_envs * self.num_ilqr_iters))
        # # logger.dumpkvs()

        optimized_action = self.u_array_val[0, :, :]

        # shift
        self.shift_u_array(u_new=None)

        return optimized_action, [], 0#sum_returns_diff/self.horizon

    def _run_open_loop(self, u_array, init_obs):
        returns = 0
        _ = self._env.reset_from_obs(init_obs)

        for i in range(self.horizon):
            x, reward, _, _ = self._env.step(u_array[i])
            returns += self.discount**i * reward

        return returns

    def update_u_per_env(self, idx, utils_list):
        u_array = self.u_array_val[:, idx, :]
        x_array, J_val, f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux = utils_list

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
                self._increase_mu(idx)
                backward_pass_counter += 1

        if not backward_accept:
            return J_val, None

        """
        Forward Pass (break if 0 < c_1 < z)
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
                x_prime = self.dynamics_model.predict(x[None], u[None], pred_type='mean', deterministic=True)[0]
                # # sanity check
                # x_prime_second_predict = self.dynamics_model.predict(x[None], u[None], pred_type='mean', deterministic=True)[0]
                # if not np.allclose(x_prime, x_prime_second_predict):
                #     logger.log(f'call predict twice, {x_prime}, {x_prime_second_predict}')
                #     raise RuntimeError
                reward = self._env.reward(x, u, x_prime)
                opt_J_val += -self.discount**i * reward
                x = x_prime

            delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
            if J_val > opt_J_val and J_val - opt_J_val > self.c_1 * (- delta_J_alpha):
                opt_u_array = np.stack(opt_u_array, axis=0)
                self.u_array_val[:, idx, :] = opt_u_array
                forward_accept = True
            else:
                # continue line search
                alpha /= self.alpha_decay_factor
                forward_pass_counter += 1

        if forward_accept:
            if self.verbose and idx == 0:
                # logger.log(f'accepted after {backward_pass_counter, forward_pass_counter} failed iterations, mu = {mu}')
                logger.log(f'mu = {mu}, J_val, opt_J_val, J_val-opt_J_val, -delta_J_alpha = {J_val, opt_J_val, J_val-opt_J_val, -delta_J_alpha}')

            self._decrease_mu(idx)
            return J_val, dict(mu=self.param_array[idx][0],
                               u_clipped_pct=np.sum(np.abs(opt_u_array) == np.mean(self.act_high))/(self.horizon*self.act_dim),
                               opt_J_val=opt_J_val, delta_J_alpha=delta_J_alpha)
        else:
            # if self.verbose:
                # logger.log(f'bw, fw = {backward_pass_counter, forward_pass_counter}, mu = {mu}')
                # logger.log(f'J_val = {J_val}')
            self._increase_mu(idx)
            return J_val, None

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
            raise OverflowError
        self.param_array[idx] = (mu, delta)

    def _reset_mu(self):
        self.param_array = [(self.mu_init, self.delta_init) for _ in range(self.num_envs)]

    def reset_u_array(self, u_array=None):
        # self._reset_mu()
        if u_array is None:
            u_array = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.horizon, self.num_envs, self.act_dim))
        self.u_array_val[:, :, :] = u_array  # broadcast

    def shift_u_array(self, u_new):
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
