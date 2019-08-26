from meta_mb.utils.serializable import Serializable
from meta_mb.policies.np_linear_policy import LinearPolicy
import numpy as np
import copy
from meta_mb.logger import logger
import ipopt
import time
from meta_mb.policies.ipopt_problems.ipopt_collocation_problem import IPOPTCollocationProblem
from meta_mb.policies.ipopt_problems.ipopt_shooting_problem import IPOPTShootingProblem
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
    ):
        Serializable.quick_init(self, locals())
        self.discount = discount
        self.method_str = method_str
        self.initializer_str = initializer_str
        self.horizon = horizon
        self.env = env

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_low, self.act_high = env.action_space.low, env.action_space.high  # wrapped or unwrapped?

        self._env = copy.deepcopy(env)

        if self.method_str == 'ipopt_collocation':
            self.executor = copy.deepcopy(env)
            self.get_actions_factory = self.get_actions_ipopt_collocation

            self.u_array = self._init_u_array()

            # initialize nlp problem
            self.problem_obj = IPOPTCollocationProblem(env, horizon, self.discount, eps=eps)
            problem_config = dict(
                n=(horizon - 1) * self.obs_dim + horizon * self.act_dim,
                m=(horizon - 1) * self.obs_dim,
                problem_obj=self.problem_obj,
                cl=np.zeros(((horizon-1) * self.obs_dim,)),
                cu=np.zeros(((horizon-1) * self.obs_dim,)),
                ub=np.concatenate([np.ones(((horizon-1)*self.obs_dim,)) * 1e2]
                                    + [self.act_high] * horizon),
                lb=np.concatenate([-np.ones(((horizon-1) * self.obs_dim,)) * 1e2]
                                    + [self.act_low] * horizon),
            )
            self.nlp = nlp = ipopt.problem(**problem_config)
            nlp.addOption('mu_strategy', 'adaptive')
            nlp.addOption('tol', 1e-5)
            nlp.addOption('max_iter', 100)
            nlp.addOption('derivative_test', 'first-order')  # SLOW

        elif self.method_str == 'ipopt_shooting':
            self.executor = copy.deepcopy(env)
            self.get_actions_factory = self.get_actions_ipopt_shooting

            # initialize s_array, a_array
            self.u_array = self._init_u_array()
            # self.running_s_array, returns = self._run_open_loop(self.running_a_array)
            # self.running_s_array = self.running_s_array[:-1]
            # logger.log('InitialReturn', returns)

            # initialize nlp problem
            self.problem_obj = IPOPTShootingProblem(env, self.horizon, self.discount, eps=eps)
            problem_config = dict(
                n=self.horizon * self.act_dim,
                m=0,
                problem_obj=self.problem_obj,
                lb=np.concatenate([self.env.action_space.low] * self.horizon),
                ub=np.concatenate([self.env.action_space.high] * self.horizon),
            )
            self.nlp = nlp = ipopt.problem(**problem_config)
            nlp.addOption('max_iter', 100)
            nlp.addOption('tol', 1e-5)
            nlp.addOption('mu_strategy', 'adaptive')
            # nlp.addOption('hessian_approximation', 'limited-memory')
            # nlp.addOption('derivative_test', 'first-order')  # SLOW
            # nlp.addOption('derivative_test_print_all', 'yes')

        elif self.method_str == 'ipopt_shooting_w_policy':
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
            self.nlp = nlp = ipopt.problem(**problem_config)
            nlp.addOption('max_iter', 100)
            nlp.addOption('tol', 1e-5)
            nlp.addOption('mu_strategy', 'adaptive')
            # nlp.addOption('derivative_test', 'first-order')  # SLOW

            self.policy_flatten_params = problem_obj_policy.get_param_values_flatten()

        else:
            raise NotImplementedError

    @property
    def vectorized(self):
        return True

    def get_rollouts(self, deterministic, plot_first_rollout):
        self.get_rollouts_ipopt()

    def _run_open_loop(self, a_array, init_obs):
        s_array, returns = [], 0
        obs = self._env.reset_from_obs(init_obs)

        try:
            assert np.allclose(obs, init_obs)
        except AssertionError:
            logger.log('WARNING: assertion error from reset_from_obs')

        for t in range(self.horizon):
            s_array.append(obs)
            obs, reward, _, _ = self._env.step(a_array[t])
            returns += self.discount ** t * reward
        s_array = np.stack(s_array, axis=0)

        return s_array, returns

    def _compute_collocation_loss(self, s_array, a_array):
        sum_rewards, reg_loss = 0, 0
        for t in range(self.horizon):
            _ = self._env.reset_from_obs(s_array[t])
            s_next, reward, _, _ = self._env.step(a_array[t])
            sum_rewards += self.discount ** t * reward
            if t < self.horizon - 1:
                reg_loss += np.linalg.norm(s_array[t+1] - s_next)**2
        return -sum_rewards, reg_loss

    def _init_u_array(self):
        if self.initializer_str == 'cem':
            init_u_array = self._init_u_array_cem()
        elif self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high,
                                             size=(self.horizon, self.act_dim))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.act_dim))
            init_u_array = np.clip(init_u_array, a_min=self.env.action_space.low, a_max=self.env.action_space.high)
        else:
            raise NotImplementedError
        return init_u_array

    def get_rollouts_ipopt(self):
        obs = self.executor.reset()
        sum_rewards = 0

        for t in range(self.max_path_length):
            opt_time = time.time()
            optimized_action = self.get_actions_factory(obs)
            opt_time = time.time() - opt_time
            obs, reward, _, _ = self.executor.step(optimized_action)
            sum_rewards += reward

            logger.logkv('PathLength', t)
            logger.logkv('Reward', reward)
            logger.logkv('TotalReward', sum_rewards)
            logger.logkv('OptTime', opt_time)
            logger.dumpkvs()

        return [sum_rewards]

    def get_actions_ipopt_collocation(self, obs):
        self.problem_obj.set_init_obs(obs)
        a_array = self.u_array
        s_array, returns = self._run_open_loop(a_array, obs)
        # s_array = s_array[1:]  # s_array[:-1, :]  # FIXME: which one???
        logger.log('PrevReturn', returns)

        # Feed in trajectory s[2:T], a[1:T], with s[1] == obs
        x0 = self.problem_obj.get_x(s=s_array[1:], a=a_array)
        x, info = self.nlp.solve(x0)
        s_array, a_array = self.problem_obj.get_s_a(x)
        optimized_action = a_array[0]

        # logging
        logger.logkv('Action100%', np.max(a_array, axis=None))
        logger.logkv('Action0%', np.min(a_array, axis=None))
        logger.logkv('Action75%', np.percentile(a_array, q=75, axis=None))
        logger.logkv('Action25%', np.percentile(a_array, q=25, axis=None))
        logger.logkv('Obs100%', np.max(s_array, axis=None))
        logger.logkv('Obs0%', np.min(s_array, axis=None))

        # shift
        u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.act_dim,))
        self.u_array = np.concatenate([a_array[1:, :], u_new[None]])

        # neg_return, reg_loss = self._compute_collocation_loss(np.concatenate([[obs], self.running_s_array]),
        #                                                        self.running_a_array)
        # logger.logkv('ColNegReturn', neg_return)
        # logger.logkv('ColRegLoss', reg_loss)

        return optimized_action

    def get_actions_ipopt_shooting(self, obs):
        self.problem_obj.set_init_obs(obs)
        x0 = self.problem_obj.get_x(self.u_array)
        x, info = self.nlp.solve(x0)
        a_array = self.problem_obj.get_a(x)
        optimized_action = a_array[0]

        # logging
        logger.logkv('Action100%', np.max(a_array, axis=None))
        logger.logkv('Action0%', np.min(a_array, axis=None))
        logger.logkv('Action75%', np.percentile(a_array, q=75, axis=None))
        logger.logkv('Action25%', np.percentile(a_array, q=25, axis=None))

        # shift
        u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.act_dim,))
        self.u_array = np.concatenate([a_array[1:, :], u_new[None]])

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
