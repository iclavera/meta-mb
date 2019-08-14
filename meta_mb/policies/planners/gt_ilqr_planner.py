import numpy as np
from meta_mb.logger import logger
import time
import scipy.linalg as sla
import pickle as pickle
from multiprocessing import Process, Pipe


class GTiLQRPlanner(object):
    def __init__(self, env, n_parallel, horizon, eps, u_array, reg_str='V',
                 mu_min=1e-6, mu_max=1e10, mu_init=1e-5, delta_0=2, delta_init=1.0, alpha_decay_factor=3.0,
                 c_1=1e-7, max_forward_iters=10, max_backward_iters=10,
                 forward_stop_cond='rel',
                 use_hessian_f=False, verbose=False):
        self._env = env
        self.n_parallel = n_parallel
        self.horizon = horizon
        self.u_array = u_array
        self.reg_str = reg_str
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
        # self.x_array, self.J_val = None, None

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_parallel)])
        if use_hessian_f:
            self.fn_str_array = ['jac_f_x', 'jac_f_u', 'hessian_f_xx', 'hessian_f_uu', 'hessian_f_ux']
        else:
            self.fn_str_array = ['jac_f_x', 'jac_f_u']

        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)

        self.ps = [
            Process(
                target=DDP_worker,
                args=(work_remote, remote, pickle.dumps(env), eps,
                      horizon, self.obs_dim, self.act_dim, self.act_low, self.act_high,
                      seed, verbose),
            ) for (work_remote, remote, seed) \
            in zip(self.work_remotes, self.remotes, seeds)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        logger.log(f'Planner started {self.n_parallel} workers...')

        for remote in self.work_remotes:
            remote.close()

    def _compute_deriv(self, inputs_dict):
        # for fn_str, remote in zip(self.fn_str_array, self.remotes):
        #     remote.send((fn_str, inputs_dict))
        for i, fn_str in enumerate(self.fn_str_array):
            self.remotes[i%self.n_parallel].send((fn_str, inputs_dict))

        dl = self._env.dl_dict(inputs_dict).values()
        df = [self.remotes[i%self.n_parallel].recv() for i in range(len(self.fn_str_array))]
        return dl, df

    def _f(self, x, u):
        _ = self._env.reset_from_obs(x)
        x_prime, reward, _, _ = self._env.step(u)
        return x_prime, reward

    def update_x_u_for_one_step(self, obs):
        u_array = self.u_array
        optimized_action = u_array[0]
        # if self.x_array is None:
        #     x_array, J_val = self._run_open_loop(u_array=u_array, init_obs=obs)
            # self.x_array, self.J_val = x_array, J_val
        # else:
        x_array, J_val = self._run_open_loop(u_array=u_array, init_obs=obs)
            # x_array, J_val = self.x_array, self.J_val
        backward_accept, forward_accept = False, False

        assert x_array.shape == (self.horizon, self.obs_dim)
        assert u_array.shape == (self.horizon, self.act_dim)

        """
        Derivatives
        """
        dl, df = self._compute_deriv(dict(obs=x_array, act=u_array))  # dl, df has length = horizon
        l_x, l_u, l_xx, l_uu, l_ux = dl
        if self.use_hessian_f:
            f_x, f_u, f_xx, f_uu, f_ux = df
        else:
            f_x, f_u = df
            f_xx, f_uu, f_ux = None, None, None
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
                    if self.use_hessian_f:
                        Q_uu = l_uu[i] + f_u[i].T @ V_prime_xx @ f_u[i] + np.tensordot(V_prime_x, f_uu[i], axes=1)
                    else:
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

        # self._decrease_mu()
        # logger.log(f'backward accepted after {backward_pass_counter} failed iterations, mu = {self.mu}')

        # """
        # Forward Pass (pick the best alpha after trying all)
        # This chunk is not executed.
        # """
        # if self.forward_stop_cond == 'abs':
        #     inputs_dict = dict(
        #         open_k_array=open_k_array,
        #         closed_K_array=closed_K_array,
        #         # delta_J_1=delta_J_1,
        #         # delta_J_2=delta_J_2,
        #         J_val=J_val,
        #         u_array=u_array,
        #         x_array=x_array,
        #     )
        #     alpha_array = 0.91**(np.arange(self.max_forward_iters)**2)
        #     for idx, alpha in enumerate(alpha_array):
        #         self.remotes[i%self.n_parallel].send(('forward_pass', alpha, inputs_dict))
        #
        #     results = [self.remotes[i%self.n_parallel].recv() for i in range(self.max_forward_iters)]
        #     logger.log('\n'.join([f"{result['alpha']}: {J_val - result['opt_J_val']}" for result in results if result]))
        #     results = filter(lambda d: d, results)  # filter out results with loss increment
        #     best_result = min(results, key=lambda d: d['opt_J_val'], default=dict())
        #     if best_result:
        #         forward_accept = True
        #
        #         # optimized_action = best_result['optimized_action']
        #         # opt_J_val, delta_J_alpha = best_result['opt_J_val'], best_result['delta_J_alpha']
        #         alpha, opt_J_val = best_result['alpha'], best_result['opt_J_val']
        #         # reward_array = best_result['reward_array']
        #         self.u_array = best_result['opt_u_array']
        #         optimized_action = self.u_array[0, :]
        #         # self.x_array, self.u_array = best_result['opt_x_array'], best_result['opt_u_array']
        #         # self.J_val = opt_J_val
        #
        #         delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
        #         logger.log(f'pick alpha = {alpha}, actual = {J_val - opt_J_val}, exp = {-delta_J_alpha}')
        #         if J_val - opt_J_val > self.c_1 * (- delta_J_alpha):
        #             self._decrease_mu()
        #         else:  # still accept but increase mu to stay in trust region
        #             self._increase_mu()
        #
        #         return optimized_action, backward_accept, forward_accept, (-J_val, -opt_J_val, -delta_J_alpha), None  # reward_array
        #     else:  # all results are invalid, best_result is dict()
        #         self._increase_mu()
        #         return optimized_action, backward_accept, forward_accept, None, None
        # else:
        #     assert self.forward_stop_cond == 'rel'

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
        # self.x_array, self.J_val = None, None

    def _run_open_loop(self, u_array, init_obs):
        x_array, sum_rewards = [], 0
        x = self._env.reset_from_obs(init_obs)
        try:
            assert np.allclose(x, init_obs)  # FAILS IN REACHER
        except AssertionError:
            logger.log('ERROR from reset_from_obs')

        for i in range(self.horizon):
            x_array.append(x)
            x, reward, _, _ = self._env.step(u_array[i])
            sum_rewards += reward
        x_array = np.stack(x_array, axis=0)

        return x_array, -sum_rewards


def DDP_worker(remote, parent_remote, env_pickle, eps,
                       horizon, obs_dim, act_dim, act_low, act_high,
                       seed, verbose):

    # batch_size = num_rollouts in the original env executors (number of experts
    # when the dynamics model is ground truth)

    print('DDP/iLQR worker starts...')

    parent_remote.close()

    env = pickle.loads(env_pickle)
    np.random.seed(seed)

    def f(x, u, return_reward=False):
        _ = env.reset_from_obs(x)
        x_prime, reward, _, _ = env.step(u)
        if return_reward:
            return x_prime, reward
        return x_prime

    def jac_f_x(x, u, centered=True):
        """
        :param x: (act_dim.)
        :param u: (obs_dim,)
        :return: (obs_dim, obs_dim)
        """
        jac = np.zeros((obs_dim, obs_dim))
        e_i = np.zeros((obs_dim,))
        if centered:
            for i in range(obs_dim):
                e_i[i] = eps
                jac[:, i] = (f(x+e_i, u) - f(x-e_i, u)) / (2*eps)
                e_i[i] = 0
        else:
            f_val = f(x, u)
            for i in range(obs_dim):
                e_i[i] = eps
                jac[:, i] = (f(x+e_i, u) - f_val) / eps
                e_i[i] = 0
        return jac

    def jac_f_u(x, u, centered=True):
        """
        :param x: (act_dim.)
        :param u: (obs_dim,)
        :return: (obs_dim, act_dim)
        """
        jac = np.zeros((obs_dim, act_dim))
        e_i = np.zeros((act_dim,))
        if centered:
            for i in range(act_dim):
                e_i[i] = eps
                jac[:, i] = (f(x, u+e_i) - f(x, u-e_i)) / (2*eps)
                e_i[i] = 0
        else:
            f_val = f(x, u)
            for i in range(act_dim):
                e_i[i] = eps
                jac[:, i] = (f(x, u+e_i) - f_val) / eps
                e_i[i] = 0
        return jac

    def hessian_f_xx(x, u):
        hess = np.zeros((obs_dim, obs_dim, obs_dim))
        f_val = f(x, u)
        e_i, e_j = np.zeros((obs_dim,)), np.zeros((obs_dim,))
        for i in range(obs_dim):
            e_i[i] = eps
            f_val_fix_i = f(x+e_i, u) - f_val
            for j in range(obs_dim):
                e_j[j] = eps
                hess[:, i, j] = (f(x+e_i+e_j, u) - f(x+e_j, u) - f_val_fix_i) / eps**2
                e_j[j] = 0
            e_i[i] = 0
        return (hess + np.transpose(hess, axes=[0, 2, 1])) * 0.5

    def hessian_f_uu(x, u):
        hess = np.zeros((obs_dim, act_dim, act_dim))
        f_val = f(x, u)
        e_i, e_j = np.zeros((act_dim,)), np.zeros((act_dim,))
        for i in range(act_dim):
            e_i[i] = eps
            f_val_fix_i = f(x, u+e_i) - f_val
            for j in range(act_dim):
                e_j[j] = eps
                hess[:, i, j] = (f(x, u+e_i+e_j) - f(x, u+e_j) - f_val_fix_i) / eps**2
                e_j[j] = 0
            e_i[i] = 0
        return (hess + np.transpose(hess, axes=[0, 2, 1])) * 0.5

    def hessian_f_ux(x, u):
        hess = np.zeros((obs_dim, act_dim, obs_dim))
        f_val = f(x, u)
        e_i, e_j = np.zeros((act_dim,)), np.zeros((obs_dim,))
        for i in range(act_dim):
            e_i[i] = eps
            f_val_fix_i = f(x, u+e_i) - f_val
            for j in range(obs_dim):
                e_j[j] = eps
                hess[:, i, j] = (f(x+e_j, u+e_i) - f(x, u+e_i) - f_val_fix_i) / eps**2
                e_j[j] = 0
            e_i[i] = 0
        # return (hess + np.transpose(hessian_f_xu(x, u), axes=[0, 2, 1])) * 0.5
        return hess

    def forward_pass(alpha, open_k_array, closed_K_array, J_val, u_array, x_array):
        # initialize
        # opt_x_array, opt_u_array = [], []
        opt_u_array = []
        # reward_array = []
        opt_J_val = 0
        info_dict = dict()

        # forward pass
        x = x_array[0]
        for i in range(horizon):
            u = u_array[i] + alpha * open_k_array[i] + closed_K_array[i] @ (x - x_array[i])
            u = np.clip(u, act_low, act_high)  # FIXME: clip here?

            # store updated state/action
            # opt_x_array.append(x)
            opt_u_array.append(u)
            time.sleep(0.004)
            x, reward = f(x, u, return_reward=True)
            # reward_array.append(reward)
            opt_J_val += -reward

        if J_val > opt_J_val:  # valid update

            # delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
            # optimized_action = opt_u_array[0]
            # opt_x_array, opt_u_array = np.stack(opt_x_array, axis=0), np.stack(opt_u_array, axis=0)
            opt_u_array = np.stack(opt_u_array, axis=0)
            info_dict = dict(
                alpha=alpha,
                # optimized_action=optimized_action,
                # opt_x_array=opt_x_array,
                opt_u_array=opt_u_array,
                opt_J_val=opt_J_val,
                # delta_J_alpha=delta_J_alpha,
                # reward_array=reward_array,
            )

        return info_dict

    str_to_fn = dict(jac_f_x=jac_f_x, jac_f_u=jac_f_u, hessian_f_xx=hessian_f_xx, hessian_f_uu=hessian_f_uu, hessian_f_ux=hessian_f_ux)

    while True:
        # receive command and data from the remote
        cmd, *data = remote.recv()
        # do a step in each of the environment of the worker
        if cmd == 'close':
            remote.close()
            break
        elif cmd == 'forward_pass':
            alpha, inputs_dict, = data
            result = forward_pass(alpha=alpha, **inputs_dict)
            remote.send(result)
        else:
            inputs_dict, = data
            x_array, u_array = inputs_dict['obs'], inputs_dict['act']

            try:
                fn = str_to_fn[cmd]

            except KeyError:
                print(f'receiving command {cmd}')
                raise NotImplementedError

            # result = [fn(x_array[i], u_array[i]) for i in range(horizon-1, -1, -1)]  # FIXME: WHY INVERSE ORDER????
            result = [fn(x_array[i], u_array[i]) for i in range(horizon)]
            remote.send(result)


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
