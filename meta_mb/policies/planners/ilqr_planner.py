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
        self.u_array_val = None
        self.model_idx_val = None
        self.model_idx_val_init = np.random.randint(low=0, high=self.dynamics_model.num_models, size=(horizon, num_envs))

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
        self.model_idx_ph = tf.placeholder(
            dtype=tf.int32,
            shape=(self.horizon, self.num_envs),
            name='perm',
        )

        build_time = time.time()
        self.utils_sym_by_env = self._build_deriv_graph()
        self.cem_optimized_actions = self._build_cem_graph()
        logger.log('TimeBuildGraph', build_time - time.time())

    def _build_deriv_graph(self):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        discount = self.discount
        num_envs = self.num_envs
        utils_sym_by_env = []

        for env_idx in range(num_envs):
            obs = self.obs_ph[env_idx, :]
            disc_reward = tf.zeros(())
            x_array, deriv_array = [], []
            for i in range(self.horizon):
                if self.use_hessian_f:
                    # hack in order to compute f_ux
                    act = self.u_array_ph[i, env_idx, :]
                    obs_act_concat = tf.concat([obs, act], axis=0)
                    obs, act = obs_act_concat[:obs_dim], obs_act_concat[-act_dim:]
                    model_idx = self.model_idx_ph[i, env_idx]
                    next_obs = self.dynamics_model.predict_sym(obs[None], act[None], pred_type=model_idx)[0]
                    reward = self._env.tf_reward(obs, act, next_obs)

                    # compute gradients
                    hess = utils.hessian_wrapper(next_obs, obs_act_concat, obs_dim, obs_dim+act_dim)  # (obs_dim, obs_dim+act_dim, obs_dim+act_dim)
                    f_x = utils.jacobian_wrapper(next_obs, obs, obs_dim, obs_dim)
                    f_u = utils.jacobian_wrapper(next_obs, act, obs_dim, act_dim)
                    f_xx = hess[:, :obs_dim, :obs_dim]
                    f_uu = hess[:, -act_dim:, -act_dim:]
                    f_ux = (hess[:, -act_dim:, :obs_dim] + tf.transpose(hess[:, :obs_dim, -act_dim:], perm=[0, 2, 1])) * 0.5
                    l_x, l_u, l_xx, l_uu, l_ux = self._env.tf_dl(obs, act, next_obs, f_x, f_xx)

                else:
                    # obs, act, next_obs, reward
                    act = self.u_array_ph[i, env_idx, :]
                    model_idx = self.model_idx_ph[i, env_idx]
                    next_obs = self.dynamics_model.predict_sym(obs[None], act[None], pred_type=model_idx)[0]
                    reward = self._env.tf_reward(obs, act, next_obs)

                    # compute gradients
                    f_x = utils.jacobian_wrapper(next_obs, obs, obs_dim, obs_dim)
                    f_u = utils.jacobian_wrapper(next_obs, act, obs_dim, act_dim)
                    f_xx = utils.hessian_wrapper(next_obs, obs, obs_dim, obs_dim)
                    f_uu, f_ux = None, None
                    l_x, l_u, l_xx, l_uu, l_ux = self._env.tf_dl(obs, act, next_obs, f_x, f_xx)

                # store
                disc_reward += discount**i * reward
                x_array.append(obs)
                deriv_array.append([f_x, f_u, f_xx, f_uu, f_ux, l_x, l_u, l_xx, l_uu, l_ux])
                obs = next_obs

            x_array = tf.stack(x_array, axis=0)
            deriv_array = list(map(lambda arr: tf.stack(arr, axis=0), zip(*deriv_array)))  # (horizon, *)

            utils_sym_by_env.append([x_array, -disc_reward, *deriv_array])

        return utils_sym_by_env

    def _build_cem_graph(self):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        horizon = self.horizon
        n_candidates = self.n_candidates
        num_envs = self.num_envs
        alpha = self.cem_alpha

        mean = tf.ones(shape=[horizon, num_envs, 1, act_dim]) * (self.act_high + self.act_low) / 2
        var = tf.ones(shape=[horizon, num_envs, 1, act_dim]) * (self.act_high - self.act_low) / 16

        init_obs = tf.reshape(
            tf.tile(tf.expand_dims(self.obs_ph, axis=1), [1, n_candidates, 1]),
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
                        if success:
                            logger.logkv(f'u_clipped_pct-{env_idx}', agent_info['u_clipped_pct'])
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
        self._shift(u_new=None)  # FIXME: alternative ways to initialize u_new

        return optimized_actions, []#, next_obs
    #
    # def _run_open_loop(self, u_array, init_obs):
    #     returns = 0
    #     _ = self._env.reset_from_obs(init_obs)
    #
    #     for i in range(self.horizon):
    #         x, reward, _, _ = self._env.step(u_array[i])
    #         returns += self.discount**i * reward
    #
    #     return returns

    def update_u_per_env(self, idx, utils_list):
        u_array = self.u_array_val[:, idx, :]
        model_idx = self.model_idx_val[:, idx]
        x_array, J_val, f_x, f_u, f_xx, f_uu, f_ux, l_x, l_u, l_xx, l_uu, l_ux = utils_list
        agent_info = dict(returns=-J_val)

        """
        Backward Pass
        """
        backward_accept = False
        backward_pass_counter = 0
        while not backward_accept and backward_pass_counter < self.max_backward_iters:
            # initialize
            V_prime_xx, V_prime_x = np.zeros((self.obs_dim, self.obs_dim)), np.zeros(self.obs_dim,)
            open_k_array, closed_K_array = [None] * self.horizon, [None] * self.horizon
            delta_J_1, delta_J_2 = 0, 0
            mu, _ = self.param_array[idx]

            try:
                # backward pass
                for i in range(self.horizon-1, -1, -1):
                    # compute Q
                    if self.use_hessian_f:
                        Q_uu = l_uu[i] + f_u[i].T @ V_prime_xx @ f_u[i] + np.tensordot(V_prime_x, f_uu[i], axes=1)
                    else:
                        Q_uu = l_uu[i] + f_u[i].T @ V_prime_xx @ f_u[i]
                    if self.reg_str == 'Q':
                        Q_uu_reg = Q_uu + mu * np.eye(self.act_dim)
                    elif self.reg_str == 'V':
                        Q_uu_reg = Q_uu + mu * f_u[i].T @ f_u[i]
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
                        Q_ux_reg = Q_ux + mu * f_u[i].T @ f_x[i]
                    else:
                        raise ValueError

                    # compute control matrices
                    k = - Q_uu_reg_inv @ Q_u
                    K = - Q_uu_reg_inv @ Q_ux_reg
                    # open_k_array.append(k)
                    # closed_K_array.append(K)
                    # open_k_array.insert(0, k)
                    # closed_K_array.insert(0, K)
                    open_k_array[i] = k
                    closed_K_array[i] = K
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
            return False, agent_info

        """
        Forward Pass (break if 0 < c_1 < z)
        """
        forward_accept = False
        forward_pass_counter = 0
        alpha = self.alpha_init
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
                x_prime = self.dynamics_model.predict(x[None], u[None], pred_type=model_idx[i])[0]
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
                agent_info['u_clipped_pct'] = np.sum(np.abs(opt_u_array) >= np.mean(self.act_high))/(self.horizon*self.act_dim)
                agent_info['returns'] = -opt_J_val

                forward_accept = True
            else:
                # continue line search
                alpha /= self.alpha_decay_factor
                forward_pass_counter += 1

        if forward_accept:
            # if self.verbose and idx == 0:
            #     # logger.log(f'accepted after {backward_pass_counter, forward_pass_counter} failed iterations, mu = {mu}')
            #     logger.log(f'mu = {mu}, J_val, opt_J_val, J_val-opt_J_val, -delta_J_alpha = {J_val, opt_J_val, J_val-opt_J_val, -delta_J_alpha}')
            # logger.log(f'accepted after {backward_pass_counter, forward_pass_counter} failed itrs, mu = {mu}, alpha = {alpha}')
            logger.log(f"alpha, J, opt_J, real_diff, exp_diff = {alpha, J_val, opt_J_val, J_val - opt_J_val, -delta_J_alpha}")

            self._decrease_mu(idx)
            return True, agent_info

        else:
            # if self.verbose:
                # logger.log(f'bw, fw = {backward_pass_counter, forward_pass_counter}, mu = {mu}')
                # logger.log(f'J_val = {J_val}')
            self._increase_mu(idx)
            return False, agent_info

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

    # def reset_u_array(self, u_array=None):
    #     if u_array is None:
    #         u_array = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.horizon, self.num_envs, self.act_dim))
    #     self.u_array_val[:, :, :] = u_array  # broadcast
    #     self.perm_val = self.perm_val_init

    # def _sample_u(self):
    #     return np.clip(np.random.normal(size=(self.num_envs, self.act_dim), scale=0.1), a_min=self.act_low, a_max=self.act_high)

    def _shift(self, u_new):
        #  shift u_array
        if u_new is None:
            # u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.num_envs,self.act_dim))
            # u_new = np.mean(self.u_array_val, axis=0) + np.random.normal(size=(self.num_envs, self.act_dim)) * np.std(self.u_array_val, axis=0)
            u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.num_envs, self.act_dim))
            # u_new = self._sample_u()
        self.u_array_val = np.concatenate([self.u_array_val[1:, :, :], u_new[None]])
        #  shift model index
        self.model_idx_val = np.concatenate([self.model_idx_val[1:, :], np.random.randint(self.dynamics_model.num_models, size=(1, self.num_envs))])

    def _reset(self, obs):
        if self.initializer_str == 'cem':
            sess = tf.get_default_session()
            init_u_array, = sess.run([self.cem_optimized_actions], feed_dict={self.obs_ph: obs})
        elif self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.act_low, high=self.act_high,
                                             size=(self.horizon, self.num_envs, self.act_dim))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.num_envs, self.act_dim))
            init_u_array = np.clip(init_u_array, a_min=self.act_low, a_max=self.act_high)
        else:
            raise NotImplementedError

        self.u_array_val = init_u_array
        self.model_idx_val = np.random.randint(low=0, high=self.dynamics_model.num_models, size=(self.horizon, self.num_envs))

    def warm_reset(self, u_array):
        if self.initializer_str != 'cem':
            self.u_array_val = u_array
            self.model_idx_val = self.model_idx_val_init


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
