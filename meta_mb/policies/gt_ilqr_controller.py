from meta_mb.utils.serializable import Serializable
from meta_mb.samplers.vectorized_env_executor import ParallelDDPExecutor
import numpy as np
from meta_mb.logger import logger
import copy
import time


class GTiLQRController(Serializable):
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
        self._env = copy.deepcopy(env)
        self.planner = ParallelDDPExecutor(env, n_parallel, horizon, eps, self._init_u_array(), verbose=verbose)
        self.executor = copy.deepcopy(env)

    @property
    def vectorized(self):
        return True

    def get_rollouts(self, deterministic, plot_first_rollout):
        self.get_rollouts_ilqr()

    def get_rollouts_ilqr(self):
        obs = self.executor.reset()
        sum_rewards = 0

        for t in range(self.max_path_length):
            time_opt = time.time()
            optimized_action = self.get_actions_ilqr(obs=obs)
            time_opt = time.time() - time_opt
            obs, reward, _, _ = self.executor.step(optimized_action)
            sum_rewards += reward

            logger.logkv('Summ-PathLength', t)
            logger.logkv('Summ-Reward', reward)
            logger.logkv('Summ-TotalReward', sum_rewards)
            logger.logkv('Summ-TimeOpt', time_opt)
            # logger.dumpkvs()

        return [sum_rewards]

    def get_actions_ilqr(self, obs):
        ilqr_itr_counter = 0
        while ilqr_itr_counter < self.num_ddp_iters:
            optimized_action, backward_accept, forward_accept, planner_returns_log, reward_array = self.planner.update_x_u_for_one_step(obs=obs)
            if not backward_accept:
                # reset
                logger.log(f'perturb u array, restart ilqr iterations')
                self.planner.perturb_u_array()
                ilqr_itr_counter = 0
            else:
                # log
                if ilqr_itr_counter % 10 == 0:
                    # logger.log(f'stats for x, max = {np.max(self.planner.x_array)}, min = {np.min(self.planner.x_array)}, mean {np.mean(self.planner.x_array)}')
                    logger.log(f'stats for u, max = {np.max(self.planner.u_array)}, min = {np.min(self.planner.u_array)}, mean {np.mean(self.planner.u_array)}')

                if forward_accept:
                    old_returns, new_returns, diff = planner_returns_log
                    if reward_array is not None:
                        reward_array = np.reshape(reward_array, (-1, 5))
                        for row in reward_array:
                            logger.log(row)
                    logger.logkv('Itr', ilqr_itr_counter)
                    logger.logkv('PlannerPrevReturn', old_returns)
                    logger.logkv('PlannerReturn', new_returns)
                    logger.logkv('ExpectedDiff', diff)
                    logger.logkv('ActualDiff', new_returns - old_returns)
                    logger.dumpkvs()

                ilqr_itr_counter += 1

        # shift
        u_new = None  # self._generate_new_u()
        self.planner.shift_u_array(u_new)

        return optimized_action

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
