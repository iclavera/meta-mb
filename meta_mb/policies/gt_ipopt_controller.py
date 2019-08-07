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
            name,
            env,
            dynamics_model,
            eps,
            num_rollouts=None,
            reward_model=None,
            discount=1,
            lmbda=1,
            method_str='opt_policy',
            n_parallel=1,
            dyn_pred_str='rand',
            initializer_str='uniform',
            reg_coef=1,
            reg_str=None,
            n_candidates=1024,
            horizon=10,
            max_path_length=200,
            num_cem_iters=8,
            num_opt_iters=8,
            num_collocation_iters=500,
            num_ddp_iters=100,
            mu=1e-3,
            persistency=0.99,
            opt_learning_rate=1e-3,
            clip_norm=-1,
            percent_elites=0.1,
            use_reward_model=False,
            alpha=0.25,
            num_particles=1,
            verbose=True,
    ):
        Serializable.quick_init(self, locals())
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.method_str = method_str
        self.dyn_pred_str = dyn_pred_str
        self.initializer_str = initializer_str
        self.reg_coef = reg_coef
        assert 0 <= self.reg_coef <= 1
        self.reg_str = reg_str
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.num_cem_iters = num_cem_iters
        self.num_opt_iters = num_opt_iters
        self.num_collocation_iters = num_collocation_iters
        self.num_ddp_iters = num_ddp_iters
        self.persistency= persistency
        self.opt_learning_rate = opt_learning_rate
        self.eps = eps
        self.num_envs = num_rollouts
        self.percent_elites = percent_elites
        self.num_elites = max(int(percent_elites * n_candidates), 1)
        self.env = env
        self.use_reward_model = use_reward_model
        self.alpha = alpha
        self.num_particles = num_particles
        self.clip_norm = clip_norm
        self.lmbda = lmbda

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        self.act_low, self.act_high = env.action_space.low, env.action_space.high  # wrapped or unwrapped?

        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1
        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"
        self._env = copy.deepcopy(env)

        if self.method_str == 'ipopt_collocation':
            self.executor = copy.deepcopy(env)
            self.get_actions_factory = self.get_actions_ipopt_collocation

            self.running_a_array = self._init_u_array()

            # initialize nlp problem
            self.problem_obj = IPOPTCollocationProblem(env, self.horizon, self.discount, eps=eps)
            problem_config = dict(
                n=(self.horizon - 1) * self.obs_space_dims + self.horizon * self.action_space_dims,
                m=(self.horizon - 1) * self.obs_space_dims,
                problem_obj=self.problem_obj,
                cl=np.zeros(((self.horizon-1) * self.obs_space_dims,)),
                cu=np.zeros(((self.horizon-1) * self.obs_space_dims,)),
                ub=np.concatenate([np.ones(((self.horizon-1)*self.obs_space_dims,)) * 1e2]
                                    + [self.act_high] * self.horizon),
                lb=np.concatenate([-np.ones(((self.horizon-1) * self.obs_space_dims,)) * 1e2]
                                    + [self.act_low] * self.horizon),
            )
            self.nlp = nlp = ipopt.problem(**problem_config)
            nlp.addOption('mu_strategy', 'adaptive')
            nlp.addOption('tol', 1e-5)
            nlp.addOption('max_iter', 30)
            nlp.addOption('derivative_test', 'first-order')  # SLOW

        elif self.method_str == 'ipopt_shooting':
            self.executor = copy.deepcopy(env)
            self.get_actions_factory = self.get_actions_ipopt_shooting

            # initialize s_array, a_array
            self.running_a_array = self._init_u_array()
            # self.running_s_array, returns = self._run_open_loop(self.running_a_array)
            # self.running_s_array = self.running_s_array[:-1]
            # logger.log('InitialReturn', returns)

            # initialize nlp problem
            self.problem_obj = IPOPTShootingProblem(env, self.horizon, self.discount, eps=eps)
            problem_config = dict(
                n=self.horizon * self.action_space_dims,
                m=0,
                problem_obj=self.problem_obj,
                lb=np.concatenate([self.env.action_space.low] * self.horizon),
                ub=np.concatenate([self.env.action_space.high] * self.horizon),
            )
            self.nlp = nlp = ipopt.problem(**problem_config)
            nlp.addOption('max_iter', 30)
            nlp.addOption('tol', 1e-5)
            nlp.addOption('mu_strategy', 'adaptive')
            # nlp.addOption('hessian_approximation', 'limited-memory')
            # nlp.addOption('derivative_test', 'first-order')  # SLOW
            # nlp.addOption('derivative_test_print_all', 'yes')

        elif self.method_str == 'ipopt_shooting_w_policy':
            self.executor = copy.deepcopy(env)
            self.get_actions_factory = self.get_actions_ipopt_shooting_w_policy
            self._policy = LinearPolicy(obs_dim=self.obs_space_dims, action_dim=self.action_space_dims, output_nonlinearity=None)

            # initialize nlp problem
            problem_obj_policy = LinearPolicy(obs_dim=self.obs_space_dims, action_dim=self.action_space_dims, output_nonlinearity=None)
            self.problem_obj = IPOPTShootingProblemWPolicy(env, self.horizon, self.discount, policy=problem_obj_policy, eps=self.eps)
            problem_config = dict(
                n=self._policy.flatten_dim,
                m=self.horizon * self.action_space_dims,
                problem_obj=self.problem_obj,
                lb=np.ones((self._policy.flatten_dim)) * (-1e10),
                ub=np.ones((self._policy.flatten_dim)) * (1e10),
                cl=np.concatenate([self.act_low] * self.horizon),
                cu=np.concatenate([self.act_high] * self.horizon),
            )
            self.nlp = nlp = ipopt.problem(**problem_config)
            nlp.addOption('max_iter', 100)
            nlp.addOption('tol', 1e-5)
            nlp.addOption('mu_strategy', 'adaptive')

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
                                                     size=(self.horizon, self.action_space_dims))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.action_space_dims))
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
        a_array = self.running_a_array
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
        u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.action_space_dims,))
        self.running_a_array = np.concatenate([a_array[1:, :], u_new[None]])

        # neg_return, reg_loss = self._compute_collocation_loss(np.concatenate([[obs], self.running_s_array]),
        #                                                        self.running_a_array)
        # logger.logkv('ColNegReturn', neg_return)
        # logger.logkv('ColRegLoss', reg_loss)

        return optimized_action

    def get_actions_ipopt_shooting(self, obs):
        self.problem_obj.set_init_obs(obs)
        x0 = self.problem_obj.get_x(self.running_a_array)
        x, info = self.nlp.solve(x0)
        a_array = self.problem_obj.get_a(x)
        optimized_action = a_array[0]

        # logging
        logger.logkv('Action100%', np.max(a_array, axis=None))
        logger.logkv('Action0%', np.min(a_array, axis=None))
        logger.logkv('Action75%', np.percentile(a_array, q=75, axis=None))
        logger.logkv('Action25%', np.percentile(a_array, q=25, axis=None))

        # shift
        u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.action_space_dims,))
        self.running_a_array = np.concatenate([a_array[1:, :], u_new[None]])

        return optimized_action

    def get_actions_ipopt_shooting_w_policy(self, obs):
        self.problem_obj.set_init_obs(obs)
        x0 = self._policy.get_param_values_flatten()
        x, info = self.nlp.solve(x0)
        self._policy.set_param_values_flatten(x)
        optimized_action, _ = self._policy.get_action(obs)

        # logging
        logger.log(f'stats for optimal_action: max = {np.max(optimized_action)}, min = {np.min(optimized_action)}')
        optimized_action = np.clip(optimized_action, a_min=self.act_low, a_max=self.act_high)

        return optimized_action

    def _init_u_array_cem(self):
        assert self.num_envs == 1
        # _env = IterativeEnvExecutor(self._env, self.num_envs*self.n_candidates, self.max_path_length)
        mean = np.ones(shape=(self.horizon, self.num_envs, 1, self.action_space_dims)) \
               * (self.env.action_space.high + self.env.action_space.low) / 2
        std = np.ones(shape=(self.horizon, self.num_envs, 1, self.action_space_dims)) \
              * (self.env.action_space.high - self.env.action_space.low) / 4

        for itr in range(10):
            lb_dist, ub_dist = mean - self.env.action_space.low, self.env.action_space.high - mean
            # constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            # std = np.sqrt(constrained_var)
            std = np.minimum(lb_dist/2, ub_dist/2, std)
            act = mean + np.random.normal(size=(self.horizon, self.num_envs, self.n_candidates, self.action_space_dims)) * std
            act = np.clip(act, self.env.action_space.low, self.env.action_space.high)
            act = np.reshape(act, (self.horizon, self.num_envs*self.n_candidates, self.action_space_dims))

            returns = np.zeros((self.num_envs*self.n_candidates,))
            for idx_cand in range(self.n_candidates):
                _returns = 0
                _ = self._env.reset()
                for t in range(self.horizon):
                    _, reward, _, _ = self._env.step(act[t, idx_cand, :])
                    _returns += self.discount ** t * np.asarray(reward)
                returns[idx_cand] = _returns

            returns = np.reshape(returns, (self.num_envs, self.n_candidates))
            logger.log(np.max(returns[0], axis=-1), np.min(returns[0], axis=-1))
            act = np.reshape(act, (self.horizon, self.num_envs, self.n_candidates, self.action_space_dims))
            elites_idx = np.argsort(-returns, axis=-1)[:, :self.num_elites]  # (num_envs, n_candidates)
            elites_actions = np.stack([act.transpose((1, 2, 0, 3))[i, elites_idx[i]] for i in range(self.num_envs)])
            elites_actions = elites_actions.transpose((2, 0, 1, 3))
            mean = mean * self.alpha + np.mean(elites_actions, axis=2, keepdims=True) * (1-self.alpha)
            std = std * self.alpha + np.std(elites_actions, axis=2, keepdims=True) * (1-self.alpha)

        a_array = mean[:, 0, 0, :]
        return a_array

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
