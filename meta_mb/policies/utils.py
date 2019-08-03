import numpy as np


class IPOPTCollocationProblem(object):
    def __init__(self, env, horizon, discount, init_obs, eps=1e-6):
        self.env = env
        self.horizon = horizon
        self.discount = discount
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(np.prod(env.action_space.shape))
        self.init_obs = init_obs
        self.eps = eps

    def get_s_a(self, x):
        s = x[:(self.horizon-1) * self.obs_dim].reshape(self.horizon-1, self.obs_dim)
        a = x[(self.horizon-1) * self.obs_dim:].reshape(self.horizon, self.act_dim)
        return s, a

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        s, a = self.get_s_a(x)
        s_prev = np.concatenate([self.init_obs[None], s], axis=0).copy()
        returns = 0
        for t in range(self.horizon):
            _ = self.env.reset_from_obs(s_prev[t])
            _, reward, done, _ = self.env.step(a[t])
            returns += self.discount ** t * reward
            if done:
                break
        return -returns

    def constraints(self, x):
        s, a = self.get_s_a(x)
        s_targets = np.zeros((self.horizon-1, self.obs_dim))
        s_prev = np.concatenate([self.init_obs[None], s[:-1]], axis=0).copy()
        for t in range(self.horizon-1):
            _ = self.env.reset_from_obs(s_prev[t])
            n_s, *_ = self.env.step(a[t])
            s_targets[t] = n_s
        constraints = (s - s_targets).reshape(-1)
        return constraints

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        s_array, a_array = self.get_s_a(x)
        env = self.env
        obs_dim = self.obs_dim
        act_dim = self.act_dim
        discount = self.discount
        horizon = self.horizon

        def dr_ds(s, a):
            return env.deriv_reward_obs(obs=s, acts=a)

        def dr_da(s, a):
            return env.deriv_reward_act(obs=s, acts=a)
        #

        grad_s_stacked, grad_a_stacked = np.zeros((horizon-1, obs_dim)), \
                                         np.zeros((horizon, act_dim))

        for t in range(horizon):
            if t == 0:
                a = a_array[t]
                _grad_a = -discount ** t * dr_da(self.init_obs, a)
                grad_a_stacked[t, :] = _grad_a
            else:
                a = a_array[t]
                s = s_array[t-1]
                _grad_s = -discount ** t * dr_ds(s, a)
                _grad_a = -discount ** t * dr_da(s, a)
                grad_a_stacked[t, :] = _grad_a
                grad_s_stacked[t-1, :] = _grad_s

        grad = np.concatenate([grad_s_stacked.reshape(-1), grad_a_stacked.reshape(-1)])
        return grad

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        s_array, a_array = self.get_s_a(x)
        env = self.env
        obs_dim = self.obs_dim
        act_dim = self.act_dim
        eps = self.eps
        horizon = self.horizon

        def step_wrapper(s, a):
            _ = env.reset_from_obs(s)
            s_next, _, _, _ = env.step(a)
            return s_next

        def df_ds(s, a):  # s must not be owned by multiple workers
            """
            :param s: (act_dim,)
            :param a: (obs_dim,)
            :return: (obs_dim, obs_dim)
            """
            grad_s = np.zeros((obs_dim, obs_dim))
            for idx in range(obs_dim):  # compute grad[:, idx]
                s[idx] += eps
                right_s_next = step_wrapper(s, a)
                s[idx] -= 2 * eps
                left_s_next = step_wrapper(s, a)
                grad_s[:, idx] = (right_s_next - left_s_next) / (2 * eps)
                s[idx] += eps

            return grad_s

        def df_da(s, a):
            """
            :param s: (act_dim.)
            :param a: (obs_dim,)
            :return: (obs_dim, act_dim)
            """
            grad_a = np.zeros((obs_dim, act_dim))
            for idx in range(act_dim):  # compute grad[:, idx]
                a[idx] += eps
                right_s_next = step_wrapper(s, a)
                a[idx] -= 2 * eps
                left_s_next = step_wrapper(s, a)
                grad_a[:, idx] = (right_s_next - left_s_next) / (2 * eps)
                a[idx] += eps
            return grad_a

        # compute dl_ds
        jacob_s_stacked = []
        jacob_a_stacked = []
        s_array = np.concatenate([self.init_obs[None], s_array[:-1]], axis=0).copy()

        for t in range(horizon-1):
            jacob_c_s = np.zeros((obs_dim, obs_dim * (horizon-1)))
            jacob_c_a = np.zeros((obs_dim, act_dim * horizon))
            s, a = s_array[t], a_array[t]
            if t == 0:
                non_zero_jacob_c_s = np.eye(obs_dim)
                jacob_c_s[:, t * obs_dim:(t+1) * obs_dim] = non_zero_jacob_c_s
            else:
                non_zero_jacob_c_s = np.concatenate([-df_ds(s, a), np.eye(obs_dim)], axis=-1)
                jacob_c_s[:, (t-1) * obs_dim:(t+1) * obs_dim] = non_zero_jacob_c_s
            non_zero_jacob_c_a = -df_da(s, a)
            jacob_c_a[:, t * act_dim:(t+1) * act_dim] = non_zero_jacob_c_a
            jacob_s_stacked.append(jacob_c_s)
            jacob_a_stacked.append(jacob_c_a)
        jacob_s = np.concatenate(jacob_s_stacked, axis=0)
        jacob_a = np.concatenate(jacob_a_stacked, axis=0)
        jacobian = np.concatenate([jacob_s, jacob_a], axis=-1)
        return jacobian


class IPOPTShootingProblem(object):
    def __init__(self, env, horizon, discount, init_obs, eps=1e-6):
        self.env = env
        self.horizon = horizon
        self.discount = discount
        self.act_dim = int(np.prod(env.action_space.shape))
        self.init_obs = init_obs
        self.eps = eps

    def get_a(self, x):
        return x.reshape(self.horizon, self.act_dim)

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        returns = 0
        acts = self.get_a(x)
        _ = self.env.reset_from_obs(self.init_obs)
        for t in range(self.horizon):
            _, reward, done, _ = self.env.step(acts[t])
            returns += self.discount ** t * reward
            if done:
                break
        return -returns

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        env = self.env
        act_dim = self.act_dim
        discount = self.discount
        horizon = self.horizon
        eps = self.eps

        def step_wrapper(s, a):
            _ = env.reset_from_obs(s)
            s_next, _, _, _ = env.step(a)
            return s_next

        def return_wrapper(s, a, horizon):
            _ = env.reset_from_obs(s)
            returns = 0
            for t in range(horizon):
                _, r, _, _ = env.step(a[t])
                returns += discount ** t * r
            return returns

        def dreturns_da(s, a, horizon):
            """
            :param s: (act_dim.)
            :param a: (obs_dim,)
            :return: (obs_dim, act_dim)
            """
            old_returns = return_wrapper(s, a, horizon)
            grad_a = np.zeros((1, act_dim))
            for idx in range(act_dim):  # compute grad[:, idx]
                a[0, idx] += eps
                new_returns = return_wrapper(s, a, horizon)
                a[0, idx] -= eps
                grad_a[:, idx] = (new_returns - old_returns) / eps
            return -grad_a

        grad_a_stacked = np.zeros((horizon, act_dim))
        acts = self.get_a(x)
        env.reset_from_obs(self.init_obs)
        s = self.init_obs

        for t in range(horizon):
            grad_a_t = dreturns_da(s, acts[t:], horizon-t)
            grad_a_stacked[t] = grad_a_t
            s = step_wrapper(s, acts[t])
        return grad_a_stacked.reshape(-1)
