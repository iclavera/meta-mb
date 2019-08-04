from meta_mb.utils.serializable import Serializable
from math import ceil
from itertools import accumulate
from meta_mb.samplers.vectorized_env_executor import *
from meta_mb.policies.np_linear_policy import LinearPolicy
from meta_mb.optimizers.gt_optimizer import GTOptimizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from meta_mb.logger import logger
import os
import ipopt
from meta_mb.policies.utils import IPOPTShootingProblem, IPOPTCollocationProblem


class GTMPCController(Serializable):
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

        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1
        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        if self.method_str == 'opt_policy':
            self.horizon = horizon = max_path_length
            self._env = copy.deepcopy(env)
            self._policy = LinearPolicy(obs_dim=self.obs_space_dims, action_dim=self.action_space_dims, output_nonlinearity=np.tanh)
            self.planner = ParallelPolicyGradUpdateExecutor(
                env, n_parallel, num_rollouts, horizon, eps,
                opt_learning_rate, num_opt_iters,
                discount, verbose,
            )
            self.get_rollouts_factory = self.get_rollouts_opt_policy

        elif self.method_str == 'opt_act':
            self.horizon = horizon = max_path_length
            self._env = copy.deepcopy(env)
            if initializer_str == 'uniform':
                self.tau_mean_val = np.random.uniform(
                    low=self.env.action_space.low,
                    high=self.env.action_space.high,
                    size=(self.horizon, self.num_envs, self.action_space_dims),
                )
            elif initializer_str == 'zeros':
                self.tau_mean_val = np.zeros(
                    (self.horizon, self.num_envs, self.action_space_dims),
                )
                self.tau_mean_val = np.clip(np.random.normal(self.tau_mean_val, scale=0.05), self.env.action_space.low, self.env.action_space.high)
            else:
                raise NotImplementedError('initializer_str must be uniform or zeros')
            self.tau_std_val = 0.05 * np.ones((self.action_space_dims,))
            self.deriv_env = ParallelActionDerivativeExecutor(env, n_parallel, horizon, num_rollouts, eps, discount, verbose)
            self.tau_optimizer = GTOptimizer(alpha=self.opt_learning_rate)

            self.get_rollouts_factory = self.get_rollouts_opt_act

            # plotting
            # (num_opt_iters, batch_size)
            self.returns_array_first_rollout, self.grad_norm_first_rollout, self.tau_norm_first_rollout = [], [], []

        elif self.method_str == 'cem':
            # self.n_parallel = self.num_rollouts = n_parallel = num_rollouts = min(n_parallel, num_rollouts)
            assert n_parallel % num_rollouts == 0
            self.n_planner_per_env = n_parallel // num_rollouts

            # self.tau_mean_val = np.ones(shape=(self.horizon, self.num_envs, 1, self.action_space_dims)) \
            #                     * (self.env.action_space.high + self.env.action_space.low) / 2
            # self.tau_var_val = np.ones(shape=(self.horizon, self.num_envs, 1, self.action_space_dims)) \
            #                    * (self.env.action_space.high - self.env.action_space.low) / 16

            self.planner_env = ParallelEnvExecutor(env, n_parallel, num_rollouts*n_candidates, max_path_length)
            self.real_env = IterativeEnvExecutor(env, num_rollouts, max_path_length)
            self.get_rollouts_factory = self.get_rollouts_cem

        elif self.method_str == 'collocation':
            self.planner_env = copy.deepcopy(env)
            self.real_env = copy.deepcopy(env)
            self._env = copy.deepcopy(env)
            self.planner = ParallelCollocationExecutor(env, n_parallel, horizon, eps,
                                                       discount, verbose)
            # initialize s_array, a_array
            if self.initializer_str == 'uniform':
                self.running_a_array = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high,
                                            size=(self.horizon, self.action_space_dims))
                self.running_s_array, returns = self._run_open_loop(self.running_a_array)
                logger.logkv('InitialReturn', returns)
                logger.dumpkvs()
            elif self.initializer_str == 'zeros':
                self.running_a_array = np.zeros(shape=(self.horizon, self.action_space_dims))
                self.running_s_array, returns = self._run_open_loop(self.running_a_array)
                logger.logkv('InitialReturn', returns)
                logger.dumpkvs()
            elif self.initializer_str == 'cem':
                self.running_a_array = self.initialize_w_cem()
                self.running_s_array, returns = self._run_open_loop(self.running_a_array)
                logger.log('cem initialization gives sum_rewards', returns)
            else:
                raise NotImplementedError
            # self.running_s_array = [env.goal_obs() for _ in range(self.horizon)]
            # self.running_s_array = np.stack(self.running_s_array, axis=0)

            # self.optimizer_s = GTOptimizer(alpha=self.opt_learning_rate)
            # self.optimizer_a = GTOptimizer(alpha=self.opt_learning_rate)
            self.optimizer = GTOptimizer(alpha=self.opt_learning_rate)

            self.get_rollouts_factory = self.get_rollouts_collocation

        elif self.method_str == 'ddp':
            self.planner = ParallelDDPExecutor(env, n_parallel, horizon, eps, verbose=verbose)
            self.executor = copy.deepcopy(env)
            self._env = copy.deepcopy(env)  # used in _run_open_loop
            self.get_rollouts_factory = self.get_rollouts_DDP

        elif self.method_str == 'ipopt_collocation':
            self._env = copy.deepcopy(env)
            self.real_env = copy.deepcopy(env)
            self.get_rollouts_factory = self.get_rollouts_ipopt_collocation
            self.optimizer = GTOptimizer(opt_learning_rate)
            # initialize s_array, a_array
            if self.initializer_str == 'uniform':
                self.running_a_array = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high,
                                                         size=(self.horizon, self.action_space_dims))
                self.running_s_array, returns = self._run_open_loop(self.running_a_array)
                logger.logkv('InitialReturn', returns)
                logger.dumpkvs()
            elif self.initializer_str == 'zeros':
                self.running_a_array = np.zeros(shape=(self.horizon, self.action_space_dims))
                self.running_s_array, returns = self._run_open_loop(self.running_a_array)
                logger.logkv('InitialReturn', returns)
                logger.dumpkvs()
            elif self.initializer_str == 'cem':
                self.running_a_array = self.initialize_w_cem()
                self.running_s_array, returns = self._run_open_loop(self.running_a_array)
            else:
                raise NotImplementedError
            self.running_s_array = self.running_s_array[:-1]

        elif self.method_str == 'ipopt_shooting':
            self._env = copy.deepcopy(env)
            self.real_env = copy.deepcopy(env)
            self.get_rollouts_factory = self.get_rollouts_ipopt_shooting
            self.optimizer = GTOptimizer(opt_learning_rate)
            # initialize s_array, a_array
            if self.initializer_str == 'uniform':
                self.running_a_array = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high,
                                                         size=(self.horizon, self.action_space_dims))
                self.running_s_array, returns = self._run_open_loop(self.running_a_array)
                logger.logkv('InitialReturn', returns)
                logger.dumpkvs()
            elif self.initializer_str == 'zeros':
                self.running_a_array = np.zeros(shape=(self.horizon, self.action_space_dims))
                self.running_s_array, returns = self._run_open_loop(self.running_a_array)
                logger.logkv('InitialReturn', returns)
                logger.dumpkvs()
            elif self.initializer_str == 'cem':
                self.running_a_array = self.initialize_w_cem()
                self.running_s_array, returns = self._run_open_loop(self.running_a_array)
            else:
                raise NotImplementedError
            self.running_s_array = self.running_s_array[:-1]

        elif self.method_str == 'rs':
            raise NotImplementedError
        else:
            raise NotImplementedError

        self._local_step = 0
        self.save_dir_first_rollout = os.path.join(logger.get_dir(), 'first_rollout')
        self.save_dir_info = os.path.join(logger.get_dir(), 'info')
        os.makedirs(self.save_dir_first_rollout, exist_ok=True)
        os.makedirs(self.save_dir_info, exist_ok=True)

    @property
    def vectorized(self):
        return True

    def get_rollouts(self, deterministic, plot_first_rollout):
        self.get_rollouts_factory(deterministic, plot_first_rollout)
        self._local_step += 1

    def get_rollouts_opt_policy(self, deterministic=False, plot_first_rollout=False):
        assert deterministic
        info = self.planner.do_gradient_steps()

        if plot_first_rollout:
            logger.log(info['old_return'][0, :])  # report return for the first rollout over optimization iterations
            self.plot_first_rollout()


        logger.logkv('AverageReturn', np.mean(info['old_return'][:, -1]))
        for idx, returns in enumerate(info['old_return'][:-1]):
            logger.logkv(f'Return {idx}', returns)

        # return info['old_return'][:, -1]

    def get_rollouts_opt_act_w_replanning(self, deterministic=False, plot_first_rollout=False):
        # TODO
        pass

    def _run_open_loop(self, a_array):
        s_array, returns = [], 0
        obs = self._env.reset()
        for t in range(self.horizon):
            s_array.append(obs)
            obs, reward, _, _ = self._env.step(a_array[t])
            returns += self.discount ** t * reward

        s_array_stacked = np.stack(s_array, axis=0)
        return s_array_stacked, returns

    def _compute_collocation_loss(self, s_array, a_array):
        sum_rewards, reg_loss = 0, 0
        for t in range(self.horizon):
            _ = self._env.reset_from_obs(s_array[t])
            s_next, reward, _, _ = self._env.step(a_array[t])
            sum_rewards += self.discount ** t * reward
            if t < self.horizon - 1:
                reg_loss += np.linalg.norm(s_array[t+1] - s_next)**2
        return -sum_rewards, reg_loss

    def get_rollouts_DDP(self, deterministic=False, plot_first_rollout=False):
        _ = self.executor.reset()
        self._reset_x_u()
        sum_rewards = 0
        for t in range(self.max_path_length):
            optimized_action = self.get_actions_DDP()
            self._shift_x_u_by_one()
            _, reward, _, _ = self.executor.step(optimized_action)
            sum_rewards += reward
            logger.log(f'reward at path length {t}: {reward}')
            logger.log(f'total reward at path length {t}: {sum_rewards}')

        return [sum_rewards]

    def get_actions_DDP(self):
        # do gradient steps
        while True:
            try:
                for itr in range(self.num_ddp_iters):
                    assert self.planner.update_x_u_for_one_step()

                    if itr % 10 == 0:
                        print(f'stats for x, max = {np.max(self.planner.x_array)}, min = {np.min(self.planner.x_array)}, mean {np.mean(self.planner.x_array)}')
                        print(f'stats for u, max = {np.max(self.planner.u_array)}, min = {np.min(self.planner.u_array)}, mean {np.mean(self.planner.u_array)}')

                    # report performance
                    logger.logkv('Itr', itr)
                    logger.logkv('PlannerReturn', self.planner.compute_traj_returns())
                    logger.dumpkvs()
                break
            except AssertionError:
                self.planner.reset_x_u(init_u_array=None)

        # save optimized action BEFORE SHIFTING
        optimized_action = self.planner.u_array[0, :]
        return optimized_action

    def _reset_x_u(self):
        if self.initializer_str == 'cem':
            init_u_array = self.initialize_w_cem()
        elif self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high,
                                                     size=(self.horizon, self.action_space_dims))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.action_space_dims))
            init_u_array = np.clip(init_u_array, a_min=self.env.action_space.low, a_max=self.env.action_space.high)
        else:
            raise NotImplementedError

        self.planner.reset_x_u(init_u_array)

    def _shift_x_u_by_one(self):
        # rotate running s_array, a_array
        if self.initializer_str == 'uniform':
            u_new = np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high, size=(self.action_space_dims))
        elif self.initializer_str == 'zeros':
            u_new = np.zeros((self.action_space_dims,))
        else:
            raise NotImplementedError
        self.planner.shift_x_u_by_one(u_new)

    def get_rollouts_ipopt_collocation(self, deterministic=False, plot_first_rollout=False):
        obs = self.real_env.reset()
        sum_rewards = 0
        for t in range(self.max_path_length):
            optimized_action = self.get_actions_ipopt_collocation(obs)
            obs, reward, _, _ = self.real_env.step(optimized_action)
            sum_rewards += reward
            logger.log(f'reward at path length {t}: {reward}')
            logger.log(f'total reward at path length {t}: {sum_rewards}')

        return [sum_rewards]

    def get_rollouts_ipopt_shooting(self, deterministic=False, plot_first_rollout=False):
        obs = self.real_env.reset()
        sum_rewards = 0
        for t in range(self.max_path_length):
            optimized_action = self.get_actions_ipopt_shooting(obs)
            obs, reward, _, _ = self.real_env.step(optimized_action)
            sum_rewards += reward
            logger.log(f'reward at path length {t}: {reward}')
            logger.log(f'total reward at path length {t}: {sum_rewards}')

        return [sum_rewards]

    def get_actions_ipopt_collocation(self, obs):
        problem = IPOPTCollocationProblem(self._env, self.horizon, self.discount, obs, eps=self.eps)
        a_array, s_array = self.running_a_array, self.running_s_array
        x0 = np.concatenate([s_array.reshape(-1), a_array.reshape(-1)])
        cl = np.zeros(((self.horizon-1) * self.obs_space_dims,))
        cu = np.zeros(((self.horizon - 1) * self.obs_space_dims,))
        ub = np.concatenate([np.ones(((self.horizon-1)*self.obs_space_dims,)) * 1e3]
                            + [self.env.action_space.high] * self.horizon)
        lb = np.concatenate([-np.ones(((self.horizon - 1) * self.obs_space_dims,)) * 1e3]
                            + [self.env.action_space.low] * self.horizon)
        nlp = ipopt.problem(
            n=(self.horizon - 1) * self.obs_space_dims + self.horizon * self.action_space_dims,
            m=(self.horizon - 1) * self.obs_space_dims,
            problem_obj=problem,
            cl=cl,
            cu=cu,
            lb=lb,
            ub=ub,
        )
        nlp.addOption('mu_strategy', 'adaptive')
        nlp.addOption('tol', 1e-7)
        nlp.addOption('max_iter', 500)
        x, info = nlp.solve(x0)
        states, actions = problem.get_s_a(x)
        u_new = np.random.uniform(low=self.env.action_space.low,
                                  high=self.env.action_space.high,
                                  size=self.action_space_dims)
        self.running_a_array = np.concatenate([actions[1:], [u_new]])
        self.running_s_array = np.concatenate([states[1:], states[-1][None]])
        return actions[0]

    def get_actions_ipopt_shooting(self, obs):
        problem = IPOPTShootingProblem(self._env, self.horizon, self.discount, obs, eps=self.eps)
        a_array, s_array = self.running_a_array, self.running_s_array
        x0 = a_array.reshape(-1)
        ub = np.concatenate([self.env.action_space.high] * self.horizon)
        lb = np.concatenate([self.env.action_space.low] * self.horizon)
        nlp = ipopt.problem(
            n=self.horizon * self.action_space_dims,
            m=0,
            problem_obj=problem,
            lb=lb,
            ub=ub,
        )
        nlp.addOption('max_iter', 100)
        nlp.addOption('tol', 1e-5)
        nlp.addOption('mu_strategy', 'adaptive')
        # nlp.addOption('hessian_approximation', 'limited-memory')
        x, info = nlp.solve(x0)
        actions = x.reshape(self.horizon, self.action_space_dims)
        u_new = np.random.uniform(low=self.env.action_space.low,
                                  high=self.env.action_space.high,
                                  size=self.action_space_dims)
        self.running_a_array = np.concatenate([actions[1:], [u_new]])
        return actions[0]

    def get_rollouts_collocation(self, deterministic=False, plot_first_rollout=False):
        _ = self.real_env.reset()
        sum_rewards = 0
        for t in range(self.max_path_length):
            optimized_action = self.get_actions_collocation()
            _, reward, _, _ = self.real_env.step(optimized_action)
            sum_rewards += reward
            logger.log(f'reward at path length {t}: {reward}')
            logger.log(f'total reward at path length {t}: {sum_rewards}')

        return [sum_rewards]

    def get_actions_collocation(self):
        a_array, s_array = self.running_a_array, self.running_s_array
        loss1, loss2 = self._compute_collocation_loss(s_array=s_array, a_array=a_array)
        logger.log('Initial loss', loss1, loss2)
        # s_array, returns = self._run_open_loop(a_array)
        # logger.log('initial returns', returns)

        # do gradient steps
        # rolling_returns_average, rolling_returns_average_prev = None, None
        # returns_prev = init_returns = returns
        lmbda = self.lmbda
        for itr in range(self.num_collocation_iters):
            grad_s, grad_a = self.planner.do_gradient_steps(s_array_stacked=s_array, a_array_stacked=a_array, lmbda=lmbda)

            # delta_s_a_concat = self.optimizer.compute_delta_var(np.concatenate([grad_s, grad_a], axis=1))
            # delta_s = self.optimizer_s.compute_delta_var(grad_s)
            # delta_s, delta_a = delta_s_a_concat[:, :self.obs_space_dims], delta_s_a_concat[:, self.obs_space_dims:]
            delta_s, delta_a = self.opt_learning_rate * grad_s, self.opt_learning_rate * grad_a
            delta_s[0, :] = np.zeros((self.obs_space_dims,))
            s_array -= delta_s

            # delta_a = self.optimizer_a.compute_delta_var(grad_a)
            a_array -= delta_a
            a_array = np.clip(a_array, self.env.action_space.low, self.env.action_space.high)

            if itr < 10:
                print(f'stats for delta_s, max = {np.max(delta_s)}, min = {np.min(delta_s)}, mean {np.mean(delta_s)}')
                print(f'stats for delta_a, max = {np.max(delta_a)}, min = {np.min(delta_a)}, mean {np.mean(delta_a)}')
                print(f'stats for s, max = {np.max(s_array)}, min = {np.min(s_array)}, mean {np.mean(s_array)}')
                print(f'stats for a, max = {np.max(a_array)}, min = {np.min(a_array)}, mean {np.mean(a_array)}')


            # report performance
            if itr % 100 == 0:
                _, returns = self._run_open_loop(a_array)
                logger.logkv('Itr', itr)
                logger.logkv('Lmbda', lmbda)
                logger.logkv('PlannerReturn', returns)
                # loss1_prev, loss2_prev = loss1, loss2
                loss1, loss2 = self._compute_collocation_loss(s_array, a_array)
                logger.logkv('PlannerLoss1', loss1)
                logger.logkv('PlannerLoss2', loss2)
                logger.logkv('PlannerLoss', loss1 + loss2)
                logger.dumpkvs()

                # if lmbda > 1e-2 and loss1 > loss1_prev and loss2 < loss2_prev:
                #     lmbda /= 1.25  # more weights on first term
                # elif lmbda < 1e2 and loss1 < loss1_prev and loss2 > loss2_prev:
                #     lmbda *= 1.25  # more weights on second term

                # if itr >= 5000 and returns >= init_returns and returns <= returns_prev: #(returns_prev < 0 and returns < returns_prev * 1.01) or (returns_prev > 0 and returns < returns_prev * 0.99):
                #     break
                #
                # returns_prev = returns

                # if itr == 0:
                #     if returns >= 0:
                #         rolling_returns_average, rolling_returns_average_prev = returns * 2.0, returns * 1.5
                #     else:
                #         rolling_returns_average, rolling_returns_average_prev = returns / 2.0, returns / 1.5
                # rolling_returns_average_prev = rolling_returns_average
                # rolling_returns_average = self.persistency * rolling_returns_average + (1 - self.persistency) * returns
                # if rolling_returns_average < rolling_returns_average_prev:
                #     logger.log(f'early stop after {itr} iterations')
                #     break

        # save optimized action
        optimized_action = a_array[0, :]

        # rotate running s_array, a_array
        if self.initializer_str == 'uniform':
            self.running_a_array = np.concatenate([
                a_array[1:],
                np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high, size=(1, self.action_space_dims)),
            ], axis=0)
        elif self.initializer_str == 'zeros':
            self.running_a_array = np.concatenate([
                a_array[1:],
                np.zeros((1, self.action_space_dims)),
            ], axis=0)
        else:
            raise NotImplementedError

        return optimized_action

    def get_rollouts_opt_act(self, deterministic=False, plot_first_rollout=False):
        """

        :param observations:
        :param deterministic:
        :param plot_first_rollout:
        :return: rollouts = num_envs experts = (horizon==max_path_length, batch_size==num_envs, act_dims)
                 returns_array = [()] * num_opt_iters
        """
        if deterministic:
            tau = self.tau_mean_val
        else:
            tau = self.tau_mean_val + np.random.normal(size=np.shape(self.tau_mean_val)) * self.tau_std_val
            tau = np.clip(tau, a_min=self.env.action_space.low, a_max=self.env.action_space.high)

        returns_array, grad_norm_first_rollout, tau_norm_first_rollout = [], [], []
        for itr in range(self.num_opt_iters):
            grad_tau, returns = self.deriv_env.get_derivative(tau)
            tau += self.tau_optimizer.compute_delta_var(grad_tau)  # Adam optimizer
            # regularization
            # clipping and regularization needs to be modified if not (low, high) = (-1, 1)
            if self.reg_str == 'poly':
                tau = np.clip(tau, a_min=self.env.action_space.low - self.reg_coef, a_max=self.env.action_space.high + self.reg_coef)
                tau -= self.reg_coef * tau**5
            elif self.reg_str == 'scale':
                _max_abs = np.max(np.abs(tau), axis=-1)
                tau /= _max_abs
            elif self.reg_str == 'tanh':
                tau = np.tanh(tau)
            else:
                raise NotImplementedError
            returns_array.append(returns)
            grad_norm_first_rollout.append(np.linalg.norm(grad_tau[:, 0, :], axis=-1))  # list item = (horizon,)
            tau_norm_first_rollout.append(np.linalg.norm(tau[:, 0, :], axis=-1))
        returns_array = np.vstack(returns_array)  # (num_opt_iters, batch_size)

        # collect info
        self.returns_array_first_rollout.extend(returns_array[:, 0])  # [()] * trainer_iterations
        self.grad_norm_first_rollout.extend(grad_norm_first_rollout)  # [(horizon,)] * trainer_iterations
        self.tau_norm_first_rollout.extend(tau_norm_first_rollout)  # [(horizon,)] * trainer_iterations

        self.tau_mean_val = tau
        # adapt tau_std_val here

        if plot_first_rollout:
            logger.log(returns_array[:, 0])  # report return for the first rollout over optimization iterations
            self.plot_first_rollout()

            self.plot_info()

        # Do not rotate
        # if self.initializer_str == 'uniform':
        #     self.tau_mean_val = np.concatenate([
        #         tau[1:],
        #         np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high, size=(1, self.num_envs, self.action_space_dims)),
        #     ], axis=0)
        # else:
        #     self.tau_mean_val = np.concatenate([
        #         tau[1:],
        #         np.zeros((1, self.num_envs, self.action_space_dims)),
        #     ], axis=0)

        logger.logkv('AverageReturn', np.mean(returns_array[-1, :]))
        for idx, returns in enumerate(returns_array[-1, :]):
            logger.logkv(f'Return {idx}', returns)
        logger.dumpkvs()
        # return returns_array[-1, :]  # total rewards for all envs in the batch, with the latest policy

    def get_rollouts_cem(self, deterministic=True, plot_first_rollout=False):
        # FIXME: NO ROTATING YET
        _ = self.real_env.reset()
        # obs_array, reward_array, act_array, act_norm_array = [], [], [], []  # info for first env to plot
        returns = np.zeros((self.num_envs,))

        for t in range(self.max_path_length):
            # change len(env_pickled_states['envs']) from num_rollouts (pickled from real_env) to n_parallel (feed into planner_env)
            env_pickled_states = self.real_env.get_pickles()
            # same effect as np.repeat
            # env_pickled_states['envs'] = sum([[env_state for _ in range(self.n_planner_per_env)] for env_state in env_pickled_states['envs']], [])
            env_pickled_states['envs'] = np.repeat(np.asarray(env_pickled_states['envs']), self.n_planner_per_env).tolist()
            optimized_actions_mean, optimized_actions_std = self.get_actions_cem(env_pickled_states)
            if deterministic:
                actions = optimized_actions_mean
            else:
                # actions = optimized_actions_mean + np.sqrt(optimized_actions_var) * np.random.normal(size=optimized_actions_mean.shape)
                actions = optimized_actions_mean + optimized_actions_std * np.random.normal(size=optimized_actions_mean.shape)
            obs, rewards, _, _ = self.real_env.step(actions)
            returns += rewards

            logger.logkv('Reward', np.mean(rewards))
            logger.logkv('SumReward', np.mean(returns))
            logger.dumpkvs()

            # # collect info
            # obs_array.append(obs[0])
            # reward_array.append(rewards[0])
            # act_array.append(actions[0])
            # act_norm_array.append(np.linalg.norm(actions[0]))

    def initialize_w_cem(self):
        assert self.num_envs == 1
        # _env = IterativeEnvExecutor(self._env, self.num_envs*self.n_candidates, self.max_path_length)
        mean = np.ones(shape=(self.horizon, self.num_envs, 1, self.action_space_dims)) \
               * (self.env.action_space.high + self.env.action_space.low) / 2
        std = np.ones(shape=(self.horizon, self.num_envs, 1, self.action_space_dims)) \
              * (self.env.action_space.high - self.env.action_space.low) / 4

        for itr in range(2):
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

    def get_actions_cem(self, env_pickled_states):
        # mean, std = np.zeros_like(self.tau_mean_val), np.ones_like(self.tau_std_val) * 0.25
        mean = np.ones(shape=(self.horizon, self.num_envs, 1, self.action_space_dims)) \
                            * (self.env.action_space.high + self.env.action_space.low) / 2
        std = np.ones(shape=(self.horizon, self.num_envs, 1, self.action_space_dims)) \
                           * (self.env.action_space.high - self.env.action_space.low) / 4
        # returns_array = []
        for itr in range(self.num_cem_iters):
            lb_dist, ub_dist = mean - self.env.action_space.low, self.env.action_space.high - mean
            # constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            # std = np.sqrt(constrained_var)
            std = np.minimum(lb_dist/2, ub_dist/2, std)
            act = mean + np.random.normal(size=(self.horizon, self.num_envs, self.n_candidates, self.action_space_dims)) * std
            act = np.clip(act, self.env.action_space.low, self.env.action_space.high)
            act = np.reshape(act, (self.horizon, self.num_envs*self.n_candidates, self.action_space_dims))

            self.planner_env.reset_from_pickles(env_pickled_states)
            returns = np.zeros((self.num_envs*self.n_candidates,))
            for t in range(self.horizon):
                _, rewards, _, _ = self.planner_env.step(act[t])
                returns += self.discount ** t * np.asarray(rewards)

            returns = np.reshape(returns, (self.num_envs, self.n_candidates))
            logger.log(np.max(returns[0], axis=-1), np.min(returns[0], axis=-1))
            act = np.reshape(act, (self.horizon, self.num_envs, self.n_candidates, self.action_space_dims))
            elites_idx = np.argsort(-returns, axis=-1)[:, :self.num_elites]  # (num_envs, n_candidates)
            elites_actions = np.stack([act.transpose((1, 2, 0, 3))[i, elites_idx[i]] for i in range(self.num_envs)])
            elites_actions = elites_actions.transpose((2, 0, 1, 3))
            mean = mean * self.alpha + np.mean(elites_actions, axis=2, keepdims=True) * (1-self.alpha)
            std = std * self.alpha + np.std(elites_actions, axis=2, keepdims=True) * (1-self.alpha)

            # elites_returns = np.reshape(returns[elites_idx], (self.num_envs, self.num_elites))
            # returns_array.append(np.mean(elites_returns, axis=-1))  # average returns of elites_actions

            # logger.logkv('AverageReturn', np.mean(elites_returns))
            # if len(elites_returns) > 1:
            #     for idx, returns in enumerate(elites_returns):
            #         logger.logkv(f'Return {idx}', returns)
        #     if itr % 50 == 0:
        #         logger.log([returns[0] for returns in returns_array[itr-50:itr]])
        #     logger.dumpkvs()
        #
        # returns_array = np.vstack(returns_array)  # (num_opt_iters, num_envs)
        # logger.log('returns of first env throughout opt iters', returns_array[:, 0])

        # save optimized actions and rotate tau stats to prepare for planning for next steps along the path
        optimized_actions_mean = mean[0, :, 0, :]
        optimized_actions_std = std[0, :, 0, :]
        assert optimized_actions_mean.shape == (self.num_envs, self.action_space_dims)

        # rotate
        # self.tau_mean_val = np.concatenate([
        #     mean[1:],
        #     np.ones(shape=(1, self.num_envs, 1, self.action_space_dims)) \
        #     * (self.env.action_space.high + self.env.action_space.low) / 2
        # ])
        # self.tau_var_val = np.concatenate([
        #     var[1:],
        #     np.ones(shape=(1, self.num_envs, 1, self.action_space_dims)) \
        #     * (self.env.action_space.high - self.env.action_space.low) / 16
        # ])

        # return returns_array[-1, :]
        return optimized_actions_mean, optimized_actions_std

    def get_actions_rs(self, env_pickled_states):
        self.planner_env.reset_from_pickles(env_pickled_states)
        returns = np.zeros((self.num_envs*self.n_candidates,))
        act = self.get_random_action(self.horizon * self.num_envs * self.n_candidates).reshape((self.horizon,
                                                                                              self.n_candidates * self.num_envs, -1))
        cand_a = act[0].reshape((self.num_envs, self.n_candidates, -1))
        for t in range(self.horizon):
            _, rewards, _, _ = self.planner_env.step(act[t])
            returns += self.discount ** t * np.asarray(rewards)

        returns = np.reshape(returns, (self.num_envs, self.n_candidates))

        # return returns_array[-1, :]
        return cand_a[range(self.num_envs), np.argmax(returns, axis=1)], None

    def plot_first_rollout(self):
        obs_array, reward_array, act_array, act_norm_array = [], [], [], []

        if self.method_str == 'opt_policy':
            obs = self._env.reset()
            W, b = self.planner.get_param_values_first_rollout()
            self._policy.set_params(dict(W=W, b=b))

            for t in range(self.horizon):
                action, _ = self._policy.get_action(obs)
                obs, reward, _, _ = self._env.step(action)
                obs_array.append(obs)
                reward_array.append(reward)
                act_array.append(action)
                act_norm_array.append(np.linalg.norm(action))

        elif self.method_str == 'opt_act':
            _ = self._env.reset()
            act_array = self.tau_mean_val[:, 0, :]

            for act in act_array:
                obs, reward, _, _ = self._env.step(act)
                obs_array.append(obs)
                reward_array.append(reward)
                act_norm_array.append(np.linalg.norm(act))

        elif self.method_str == 'cem':
            raise NotImplementedError

        x = np.arange(self.horizon)
        obs_array = np.transpose(np.asarray(obs_array))  # (obs_dims, horizon)
        act_array = np.transpose(np.asarray(act_array))  # (act_dims, horizon)

        n_subplots = self.obs_space_dims + self.action_space_dims + 2
        nrows = ceil(np.sqrt(n_subplots))
        ncols = ceil(n_subplots/nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(70, 30))
        axes = axes.flatten()

        for i in range(self.obs_space_dims):  # split by observation space dimension
            ax = axes[i]
            ax.plot(x, obs_array[i], label=f'obs_gt')

        for i in range(self.action_space_dims):
            ax = axes[i+self.obs_space_dims]
            ax.plot(x, act_array[i], label=f'act_{i}', color='r')

        ax = axes[self.obs_space_dims+self.action_space_dims]
        # ax.plot(x, reward_array, label='reward_gt')
        ax.plot(x, act_norm_array, label='act_norm')
        ax.legend()

        ax = axes[self.obs_space_dims+self.action_space_dims+1]
        ax.plot(x, list(accumulate(reward_array)), label='reward_gt')
        # ax.plot(x, list(accumulate(loss_reward)), label='reward_planning')
        ax.legend()

        fig.suptitle(f'{self._local_step}')

        # plt.show()
        plt.savefig(os.path.join(self.save_dir_first_rollout, f'{self._local_step}.png'))
        logger.log('plt saved to', os.path.join(self.save_dir_first_rollout, f'{self._local_step}.png'))

    def plot_info(self):
        # x: iterations, y: stats average over batch
        x = np.arange(len(self.returns_array_first_rollout))  # plot every 10 steps along the path
        returns = self.returns_array_first_rollout
        # idx_horizon_array = np.arange(0, self.horizon, 10)
        grad_norm = np.stack(self.grad_norm_first_rollout, axis=-1)  # (horizon, trainer_iterations)
        tau_norm = np.stack(self.tau_norm_first_rollout, axis=-1)  # (horizon, trainer_iterations)

        self.returns_array_first_rollout = []
        self.grad_norm_first_rollout = []
        self.tau_norm_first_rollout = []

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 30))
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(x, returns)
        ax.set_xlabel('trainer_iterations')
        ax.set_ylabel('sum_rewards_for_first_rollout')

        # ax = axes[1]
        # for idx in idx_horizon_array:
        #     ax.plot(x, grad_norm[idx, :], label=idx)
        # ax.set_xlabel('trainer_iterations')
        # ax.set_ylabel('grad_norm')
        # ax.legend(title='path_length_so_far')
        #
        # ax = axes[2]
        # tau_norm[]
        # for idx in idx_horizon_array:
        #     ax.plot(x, tau_norm[idx, :], label=idx)
        # ax.set_xlabel('trainer_iterations')
        # ax.set_ylabel('avg_action_norm')
        # ax.legend(title='path_length_so_far')

        ax = axes[1]
        ax.plot(np.arange(0, self.horizon), np.mean(grad_norm, axis=-1))
        ax.set_xlabel('horizon')
        ax.set_ylabel(f'avg_grad_norm_over_{len(x)}_iteration')

        ax = axes[2]
        ax.plot(np.arange(0, self.horizon), np.mean(tau_norm, axis=-1))
        ax.set_xlabel('horizon')
        ax.set_ylabel(f'avg_action_norm_over_{len(x)}_iteration')

        fig.suptitle(f'{self._local_step}')
        fig.savefig(os.path.join(self.save_dir_info, f'{self._local_step}.png'))
        logger.log('plt saved to', os.path.join(self.save_dir_info, f'{self._local_step}.png'))

    def get_random_action(self, n):
        return np.random.uniform(low=self.env.action_space.low,
                                 high=self.env.action_space.high,
                                 size=(n,) + self.env.action_space.low.shape)

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        """
        This funciton is called by sampler at the start of each trainer iteration.
        :param dones:
        :return:
        """
        pass
        # if self.method_str == 'opt_act':
            # assert len(dones) == self.num_envs
            # if self.initializer_str == 'uniform':
            #     self.tau_mean_val = np.random.uniform(
            #         low=self.env.action_space.low,
            #         high=self.env.action_space.high,
            #         size=(self.horizon, self.num_envs, self.action_space_dims),
            #     )
            # elif self.initializer_str == 'zeros':
            #     self.tau_mean_val = np.zeros(
            #         (self.horizon, self.num_envs, self.action_space_dims),
            #     )

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
