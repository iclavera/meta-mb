import numpy as np
import copy


class IPOPTShootingProblemWPolicy(object):
    def __init__(self, env, horizon, discount, policy, eps=1e-6):
        self.env = copy.deepcopy(env)
        self.horizon = horizon
        self.discount = discount
        self.policy = policy  # stateless
        self.flatten_dim = self.policy.flatten_dim
        self.eps = eps
        self.init_obs = None
        self.act_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

    def set_init_obs(self, obs):
        self.init_obs = obs

    def get_a(self, x, obs):
        self.policy.set_param_values_flatten(x)
        obs = self.env.reset_from_obs(obs)
        act, _ = self.policy.get_action(obs)
        return act

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return self._l(x, ignore_done=True)

    def constraints(self, x):
        return self._collect_actions(x=x)

    def jacobian(self, x):
        eps = self.eps
        policy = self.policy

        jac = np.zeros((self.horizon*self.act_dim, self.flatten_dim))
        center_actions = self._collect_actions(x=x)

        flatten_idx = 0
        for i in range(self.act_dim):
            for j in range(self.obs_dim):
                policy.perturb_W(i, j, eps)
                right_actions = self._collect_actions(x=None)
                jac[:, flatten_idx] = (right_actions - center_actions) / eps
                policy.perturb_W(i, j, -eps)

                flatten_idx += 1

        for i in range(self.act_dim):
            policy.perturb_b(i, eps)
            right_actions = self._collect_actions(x=None)
            jac[:, flatten_idx] = (right_actions - center_actions) / eps
            policy.perturb_b(i, -eps)

            flatten_idx += 1

        assert flatten_idx == self.flatten_dim
        return jac

    def _collect_actions(self, x, ignore_done=True):
        if x is not None:
            self.policy.set_param_values_flatten(x)

        raw_actions_stacked = np.zeros((self.horizon, self.act_dim))
        obs = self.env.reset_from_obs(self.init_obs)

        for t in range(self.horizon):
            act, _ = self.policy.get_action(obs)
            raw_actions_stacked[t] = act
            act = np.clip(act, self.act_low, self.act_high)  # FIXME: clip here?
            obs, _, done, _ = self.env.step(act)
            if not ignore_done and done:
                break

        return raw_actions_stacked.ravel()

    def _l(self, x, ignore_done=True):
        returns = 0
        self.policy.set_param_values_flatten(x)
        obs = self.env.reset_from_obs(self.init_obs)

        for t in range(self.horizon):
            act, _ = self.policy.get_action(obs)
            act = np.clip(act, self.act_low, self.act_high)
            obs, reward, done, _ = self.env.step(act)
            returns += self.discount ** t * reward
            if not ignore_done and done:
                break

        return -returns

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        eps = self.eps
        grad = np.zeros((self.flatten_dim))
        for idx in range(self.flatten_dim):
            x[idx] += eps
            right_l = self._l(x)
            x[idx] -= 2*eps
            left_l = self._l(x)
            x[idx] += eps
            grad[idx] = (right_l - left_l) / (2 * eps)

        return grad