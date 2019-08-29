import numpy as np
import copy


class GTCollocationProblem(object):
    def __init__(self, env, horizon, discount, eps=1e-6):
        self.env = copy.deepcopy(env)
        self.horizon = horizon
        self.discount = discount
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(np.prod(env.action_space.shape))
        self.init_obs = None
        self.eps = eps

    def set_init_obs(self, obs):
        self.init_obs = obs

    def get_inputs(self, x, u):
        assert x.shape == (self.horizon-1, self.obs_dim)
        assert u.shape == (self.horizon, self.act_dim)
        return np.concatenate([x.ravel(), u.ravel()])

    def get_x_u(self, inputs):
        x_array_drop_first = inputs[:(self.horizon-1) * self.obs_dim].reshape(self.horizon-1, self.obs_dim)
        x_array = np.concatenate([self.init_obs[None], x_array_drop_first], axis=0)
        u_array = inputs[(self.horizon-1) * self.obs_dim:].reshape(self.horizon, self.act_dim)
        return x_array_drop_first, x_array, u_array

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        x_array_drop_first, x_array, u_array = self.get_x_u(x)
        returns = self.env.reward(obs=x_array, acts=u_array, next_obs=None)
        return -np.sum(returns, axis=0)
        #
        # s, a = self.get_s_a(x)
        # s_prev = np.concatenate([self.init_obs[None], s], axis=0).copy()
        # if ignore_done:
        #     returns = self.env.reward(obs=s_prev, acts=a, next_obs=None)
        #     returns = sum([self.discount**t * returns[t] for t in range(self.horizon)])
        # else:
        #     returns = 0
        #     for t in range(self.horizon):
        #         _ = self.env.reset_from_obs(s_prev[t])
        #         _, reward, done, _ = self.env.step(a[t])
        #         returns += self.discount ** t * reward
        #         if done:
        #             break
        # return -returns

    def constraints(self, x):
        x_array_drop_first, x_array, u_array = self.get_x_u(x)
        x_target_array = np.empty(shape=(self.horizon-1, self.obs_dim))
        for i in range(self.horizon-1):
            x_target_array[i] = self._step_wrapper(x=x_array[i], u=u_array[i])
        return (x_array_drop_first - x_target_array).ravel()

        # s, a = self.get_s_a(x)
        # s_targets = np.zeros((self.horizon-1, self.obs_dim))
        # s_prev = np.concatenate([self.init_obs[None], s], axis=0)#.copy()
        # for t in range(self.horizon-1):
        #     _ = self.env.reset_from_obs(s_prev[t])
        #     n_s, *_ = self.env.step(a[t])
        #     s_targets[t] = n_s
        # constraints = (s - s_targets).reshape(-1)
        # return constraints

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        x_array_drop_first, x_array, u_array = self.get_x_u(x)
        dr_dx = self.env.deriv_reward_obs(obs=x_array_drop_first, acts=u_array[1:, :])
        dr_du = self.env.deriv_reward_act(obs=x_array, acts=u_array)
        return np.concatenate([-dr_dx.ravel(), -dr_du.ravel()], axis=0)
        #
        #
        # s_array, a_array = self.get_s_a(x)
        # env = self.env
        # obs_dim = self.obs_dim
        # act_dim = self.act_dim
        # discount = self.discount
        # horizon = self.horizon
        #
        #
        # def dr_ds(s, a):
        #     return env.deriv_reward_obs(obs=s, acts=a)
        #
        # def dr_da(s, a):
        #     return env.deriv_reward_act(obs=s, acts=a)
        # #
        #
        # grad_s_stacked, grad_a_stacked = np.zeros((horizon-1, obs_dim)), \
        #                                  np.zeros((horizon, act_dim))
        #
        # for t in range(horizon):
        #     if t == 0:
        #         a = a_array[t]
        #         _grad_a = -discount ** t * dr_da(self.init_obs, a)
        #         grad_a_stacked[t, :] = _grad_a
        #     else:
        #         a = a_array[t]
        #         s = s_array[t-1]
        #         _grad_s = -discount ** t * dr_ds(s, a)
        #         _grad_a = -discount ** t * dr_da(s, a)
        #         grad_a_stacked[t, :] = _grad_a
        #         grad_s_stacked[t-1, :] = _grad_s
        #
        # grad = np.concatenate([grad_s_stacked.reshape(-1), grad_a_stacked.reshape(-1)])
        # return grad

    def _step_wrapper(self, x, u):
        _ = self.env.reset_from_obs(x)
        x_next, *_ = self.env.step(u)
        return x_next

    def _jac_f_x(self, x, u):
        """

        :param x: (obs_dim,)
        :param u: (act_dim,)
        :return:
        """
        eps = self.eps
        grad_x = np.zeros((self.obs_dim, self.obs_dim))
        for i in range(self.obs_dim):
            x[i] += eps
            right_x_next = self._step_wrapper(x, u)
            x[i] -= 2*eps
            left_x_next = self._step_wrapper(x, u)
            x[i] += eps
            grad_x[:, i] = (right_x_next - left_x_next) / (2*eps)
        return grad_x

    def _jac_f_u(self, x, u):
        """

        :param x:
        :param u:
        :return:
        """
        eps = self.eps
        grad_u = np.zeros((self.obs_dim, self.act_dim))
        for i in range(self.act_dim):
            u[i] += eps
            right_x_next = self._step_wrapper(x, u)
            u[i] -= 2*eps
            left_x_next = self._step_wrapper(x, u)
            u[i] += eps
            grad_u[:, i] = (right_x_next - left_x_next) / (2*eps)
        return grad_u

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        x_array_drop_first, x_array, u_array = self.get_x_u(x)
        obs_dim = self.obs_dim
        act_dim = self.act_dim
        horizon = self.horizon

        # def step_wrapper(s, a):
        #     _ = env.reset_from_obs(s)
        #     s_next, _, _, _ = env.step(a)
        #     return s_next
        #
        # def df_ds(s, a):  # s must not be owned by multiple workers
        #     """
        #     :param s: (act_dim,)
        #     :param a: (obs_dim,)
        #     :return: (obs_dim, obs_dim)
        #     """
        #     grad_s = np.zeros((obs_dim, obs_dim))
        #     for idx in range(obs_dim):  # compute grad[:, idx]
        #         s[idx] += eps
        #         right_s_next = step_wrapper(s, a)
        #         s[idx] -= 2 * eps
        #         left_s_next = step_wrapper(s, a)
        #         grad_s[:, idx] = (right_s_next - left_s_next) / (2 * eps)
        #         s[idx] += eps
        #
        #     return grad_s
        #
        # def df_da(s, a):
        #     """
        #     :param s: (act_dim.)
        #     :param a: (obs_dim,)
        #     :return: (obs_dim, act_dim)
        #     """
        #     grad_a = np.zeros((obs_dim, act_dim))
        #     for idx in range(act_dim):  # compute grad[:, idx]
        #         a[idx] += eps
        #         right_s_next = step_wrapper(s, a)
        #         a[idx] -= 2 * eps
        #         left_s_next = step_wrapper(s, a)
        #         grad_a[:, idx] = (right_s_next - left_s_next) / (2 * eps)
        #         a[idx] += eps
        #     return grad_a

        # compute dl_ds
        jac_c_x = np.eye((horizon-1)*obs_dim, (horizon-1)*obs_dim)
        for i in range(1, horizon-1):
            jac_c_x[i*obs_dim:(i+1)*obs_dim, (i-1)*obs_dim:i*obs_dim] = -self._jac_f_x(x=x_array[i], u=u_array[i])

        jac_c_u = np.zeros(shape=((horizon-1) * obs_dim, horizon * act_dim))
        for i in range(horizon-1):
            jac_c_u[i*obs_dim:(i+1)*obs_dim, i*act_dim:(i+1)*act_dim] = -self._jac_f_u(x=x_array[i], u=u_array[i])

        jac_c_inputs = np.concatenate([jac_c_x, jac_c_u], axis=1)
        return jac_c_inputs

        # # compute dl_ds
        # jacob_s_stacked = []
        # jacob_a_stacked = []
        # # s_array, a_array = self.get_s_a(x)
        # s_array = np.concatenate([self.init_obs[None], s_array[:-1]], axis=0).copy()
        #
        # for t in range(horizon-1):
        #     jacob_c_s = np.zeros((obs_dim, obs_dim * (horizon-1)))
        #     jacob_c_a = np.zeros((obs_dim, act_dim * horizon))
        #     s, a = s_array[t], a_array[t]
        #     if t == 0:
        #         non_zero_jacob_c_s = np.eye(obs_dim)
        #         jacob_c_s[:, t * obs_dim:(t+1) * obs_dim] = non_zero_jacob_c_s
        #     else:
        #         non_zero_jacob_c_s = np.concatenate([-df_ds(s, a), np.eye(obs_dim)], axis=-1)
        #         jacob_c_s[:, (t-1) * obs_dim:(t+1) * obs_dim] = non_zero_jacob_c_s
        #     non_zero_jacob_c_a = -df_da(s, a)
        #     jacob_c_a[:, t * act_dim:(t+1) * act_dim] = non_zero_jacob_c_a
        #     jacob_s_stacked.append(jacob_c_s)
        #     jacob_a_stacked.append(jacob_c_a)
        # jacob_s = np.concatenate(jacob_s_stacked, axis=0)
        # jacob_a = np.concatenate(jacob_a_stacked, axis=0)
        # jacobian = np.concatenate([jacob_s, jacob_a], axis=-1)
        # return jacobian
