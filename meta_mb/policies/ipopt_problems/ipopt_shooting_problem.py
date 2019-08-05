import numpy as np


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
            grad_a = np.zeros((1, act_dim))
            for idx in range(act_dim):  # compute grad[:, idx]
                a[0, idx] += eps
                right_returns = return_wrapper(s, a, horizon)
                a[0, idx] -= 2 * eps
                left_returns = return_wrapper(s, a, horizon)
                grad_a[:, idx] = (right_returns - left_returns) / (2 * eps)
                a[0, idx] += eps
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
