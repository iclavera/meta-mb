from meta_mb.utils.serializable import Serializable
from meta_mb.policies.np_linear_policy import LinearPolicy
import numpy as np
import copy
from meta_mb.logger import logger
import ipopt
from meta_mb.policies.ipopt_problems.gt_collocation_problem import GTCollocationProblem
from meta_mb.policies.ipopt_problems.gt_shooting_problem import GTShootingProblem
from meta_mb.policies.ipopt_problems.ipopt_shooting_problem_w_policy import IPOPTShootingProblemWPolicy


class GTIpoptController(Serializable):
    def __init__(
            self,
            env,
            eps,
            discount=1,
            method_str='collocation',
            initializer_str='uniform',
            horizon=10,
            n_candidates=500, 
            num_cem_iters=5,
            percent_elites=0.1,
            alpha=0.25, 
            deterministic_policy=True,
    ):
        Serializable.quick_init(self, locals())
        self.discount = discount
        self.method_str = method_str
        self.initializer_str = initializer_str
        self.horizon = horizon
        # cem
        self.n_candidates = n_candidates
        self.num_cem_iters = num_cem_iters
        self.num_elites = max(int(percent_elites * n_candidates), 1)
        self.alpha = alpha
        self.deterministic_policy = deterministic_policy

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_low, self.act_high = env.action_space.low, env.action_space.high  # wrapped or unwrapped?

        self._env = copy.deepcopy(env)

        if self.method_str == 'collocation':
            self.problem_obj = GTCollocationProblem(env, horizon, self.discount, eps=eps)
            problem_config = dict(
                n=(horizon-1) * self.obs_dim + horizon * self.act_dim,
                m=(horizon-1) * self.obs_dim,
                problem_obj=self.problem_obj,
                cl=np.zeros(((horizon-1) * self.obs_dim,)),
                cu=np.zeros(((horizon-1) * self.obs_dim,)),
                ub=np.concatenate([np.ones(((horizon-1)*self.obs_dim,)) * 1e2]
                                    + [self.act_high] * horizon),
                lb=np.concatenate([-np.ones(((horizon-1) * self.obs_dim,)) * 1e2]
                                    + [self.act_low] * horizon),
            )
        elif self.method_str == 'shooting':
            self.problem_obj = GTShootingProblem(env, self.horizon, self.discount, eps=eps)
            problem_config = dict(
                n=self.horizon * self.act_dim,
                m=0,
                problem_obj=self.problem_obj,
                lb=np.concatenate([env.action_space.low] * self.horizon),
                ub=np.concatenate([env.action_space.high] * self.horizon),
            )
        elif self.method_str == 'shooting_w_policy':
            self.executor = copy.deepcopy(env)
            self.get_actions_factory = self.get_actions_ipopt_shooting_w_policy
            # self._policy = LinearPolicy(obs_dim=self.obs_space_dims, action_dim=self.action_space_dims, output_nonlinearity=None)

            # initialize nlp problem
            problem_obj_policy = LinearPolicy(obs_dim=self.obs_dim, action_dim=self.act_dim, output_nonlinearity=None, use_filter=self.policy_filter)
            self.problem_obj = IPOPTShootingProblemWPolicy(env, self.horizon, self.discount, policy=problem_obj_policy, eps=eps)
            problem_config = dict(
                n=problem_obj_policy.flatten_dim,
                m=self.horizon * self.act_dim,
                problem_obj=self.problem_obj,
                lb=np.ones((problem_obj_policy.flatten_dim)) * (-1e10),
                ub=np.ones((problem_obj_policy.flatten_dim)) * (1e10),
                cl=np.concatenate([self.act_low] * self.horizon),
                cu=np.concatenate([self.act_high] * self.horizon),
            )
            self.policy_flatten_params = problem_obj_policy.get_param_values_flatten()
        else:
            raise NotImplementedError

        self.nlp = nlp = ipopt.problem(**problem_config)
        nlp.addOption('max_iter', 100)
        nlp.addOption('tol', 1e-3)
        nlp.addOption('mu_strategy', 'adaptive')
        # nlp.addOption('derivative_test', 'first-order')

    @property
    def vectorized(self):
        return True

    def _run_open_loop(self, a_array, init_obs):
        s_array, returns = [], 0
        obs = self._env.reset_from_obs(init_obs)

        if not np.allclose(obs, init_obs):
            logger.warn('assertion error from reset_from_obs')

        for t in range(self.horizon):
            s_array.append(obs)
            obs, reward, _, _ = self._env.step(a_array[t])
            returns += self.discount ** t * reward
        s_array = np.stack(s_array, axis=0)

        return s_array, returns

    def _init_u_array(self, obs):
        if self.initializer_str == 'cem':
            init_u_array = self._get_actions_cem(obs)
        elif self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.act_low, high=self.act_high,
                                             size=(self.horizon, self.act_dim))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.act_dim))
            init_u_array = np.clip(init_u_array, a_min=self.act_low, a_max=self.act_high)
        else:
            raise NotImplementedError
        return init_u_array
    
    def _get_actions_cem(self, obs):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        horizon = self.horizon
        n_candidates = self.n_candidates
        num_envs = 1 

        mean = np.ones(shape=(horizon, num_envs, 1, act_dim)) * (self.act_high + self.act_low) / 2
        var = np.ones(shape=(horizon, num_envs, 1, act_dim)) * (self.act_high - self.act_low) / 16

        for itr in range(self.num_cem_iters):
            lb_dist, ub_dist = mean - self.act_low, self.act_high - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            std = np.sqrt(constrained_var)
            act = mean + np.random.normal(size=(horizon, num_envs, n_candidates, act_dim)) * std
            act = np.clip(act, self.act_low, self.act_high)
            act = np.reshape(act, (horizon, num_envs*n_candidates, act_dim))
            returns = np.zeros((num_envs*n_candidates,))

            for i in range(n_candidates):
                _ = self._env.reset_from_obs(obs)
                for t in range(horizon):
                    _, reward, _, _ = self._env.step(act[t, i, :])
                    returns[i] += self.discount**t * reward

            # Re-fit belief to the best ones
            returns = np.reshape(returns, (num_envs, n_candidates))
            # logger.log(f"at cem {itr}, returns avg = {np.mean(returns, axis=1)}")
            act = np.reshape(act, (horizon, num_envs, n_candidates, act_dim))
            act = np.transpose(act, (1, 2, 0, 3))  # (num_envs, n_candidates, horizon, act_dim)
            indices = np.argsort(-returns, axis=1)[:, :self.num_elites]  # (num_envs, num_elites)
            elite_actions = np.stack([act[env_idx, indices[env_idx]] for env_idx in range(num_envs)], axis=0)
            elite_actions = np.transpose(elite_actions, (2, 0, 1, 3))  # (horizon, num_envs, n_candidates, act_dim)
            elite_mean = np.mean(elite_actions, axis=2, keepdims=True)
            elite_var = np.var(elite_actions, axis=2, keepdims=True)
            mean = mean * self.alpha + elite_mean * (1 - self.alpha)
            var = var * self.alpha + elite_var * (1 - self.alpha)

        optimized_actions_mean = mean[:, 0, 0, :]
        if self.deterministic_policy:
            optimized_actions = optimized_actions_mean
        else:
            optimized_actions_var = var[:, 0, 0, :]
            optimized_actions = mean + np.random.normal(size=np.shape(optimized_actions_mean)) * np.sqrt(optimized_actions_var)

        assert optimized_actions.shape == (horizon, act_dim)
        return optimized_actions

    def get_action(self, obs, verbose=True):
        if self.method_str == 'collocation':
            optimized_actions = self._get_actions_collocation(obs, verbose=verbose)
        elif self.method_str == 'shooting':
            optimized_actions = self._get_actions_shooting(obs, verbose=verbose)
        elif self.method_str == 'shooting_w_policy':
            self._update_policy_shooting(obs, verbose=verbose)
            optimized_actions, _ = self.stateless_policy.get_actions(obs)  # (num_envs, obs_dim)
        else:
            raise NotImplementedError

        return optimized_actions, []

    def _get_actions_collocation(self, obs, verbose):
        a_array = self._init_u_array(obs)
        s_array, returns = self._run_open_loop(a_array, obs)
        if hasattr(self._env, "get_goal_x_array"):
            s_array = self._env.get_goal_x_array(s_array)

        # Feed in trajectory s[2:T], a[1:T], with s[1] == obs
        self.problem_obj.set_init_obs(obs)
        x0 = self.problem_obj.get_x(s=s_array[1:], a=a_array)
        x, info = self.nlp.solve(x0)
        s_array, a_array = self.problem_obj.get_s_a(x)
        optimized_action = a_array[0]

        if verbose:
            logger.logkv(f'ReturnBefore', returns)
            logger.logkv(f'ReturnAfter', -info['obj_val'])
            u_clipped_pct = np.sum(np.abs(optimized_action) >= np.mean(self.act_high))/(self.horizon*self.act_dim)
            if u_clipped_pct > 0:
                logger.logkv('u_clipped_pct', u_clipped_pct)

        return optimized_action

    def _get_actions_shooting(self, obs, verbose):
        self.problem_obj.set_init_obs(obs)
        a_array = self._init_u_array(obs)
        x0 = self.problem_obj.get_x(a_array)
        x, info = self.nlp.solve(x0)
        a_array = self.problem_obj.get_a(x)
        optimized_action = a_array[0]

        if verbose:
            logger.logkv(f'ReturnAfter', -info['obj_val'])
            u_clipped_pct = np.sum(np.abs(optimized_action) >= np.mean(self.act_high))/(self.horizon*self.act_dim)
            if u_clipped_pct > 0:
                logger.logkv('u_clipped_pct', u_clipped_pct)

        return optimized_action

    def get_actions_ipopt_shooting_w_policy(self, obs):
        self.problem_obj.set_init_obs(obs)
        x0 = self.policy_flatten_params  #self._policy.get_param_values_flatten()
        x, info = self.nlp.solve(x0)
        self.policy_flatten_params = x # self._policy.set_param_values_flatten(x)
        optimized_action = self.problem_obj.get_a(x, obs=obs)  # optimized_action, _ = self._policy.get_action(obs)

        # logging
        logger.logkv('Action100%', np.max(optimized_action, axis=None))
        logger.logkv('Action0%', np.min(optimized_action, axis=None))
        optimized_action = np.clip(optimized_action, a_min=self.act_low, a_max=self.act_high)

        return optimized_action

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        """
        This funciton is called by sampler at the start of each trainer iteration.
        :param dones:
        :return:
        """
        pass

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
