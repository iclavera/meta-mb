from meta_mb.utils.serializable import Serializable
import numpy as np
from meta_mb.logger import logger
from meta_mb.policies.ipopt_problems.ipopt_dyn_collocation_problem import CollocationProblem
from meta_mb.policies.ipopt_problems.ipopt_dyn_shooting_problem import ShootingProblem
import ipopt
from meta_mb.samplers.vectorized_env_executor import IterativeEnvExecutor


class DynIpoptController(Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            num_rollouts,
            discount,
            n_parallel,
            initializer_str,
            method_str,
            n_candidates,
            horizon,
            policy=None,
            percent_elites=0.1,
            alpha=0.25,
            verbose=True,
    ):
        Serializable.quick_init(self, locals())
        self.env = env
        self.env_eval = IterativeEnvExecutor(env, num_rollouts, horizon+1)  # hack to have dones = False
        self.dynamics_model = dynamics_model
        self.discount = discount
        self.initializer_str = initializer_str
        self.method_str = method_str
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.num_envs = num_rollouts
        self.num_elites = max(int(percent_elites * n_candidates), 1)
        self.alpha = alpha

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        if self.method_str == 'ipopt_collocation':
            self.u_array_val = self._init_u_array()
            self.problem_obj = CollocationProblem(env=env, dynamics_model=dynamics_model, horizon=horizon)
            problem_config=dict(
                n=(horizon-1)*self.obs_dim + horizon*self.act_dim,
                m=(horizon-1)*self.obs_dim,
                problem_obj=self.problem_obj,
                cl=np.zeros(((horizon-1)*self.obs_dim,)),
                cu=np.zeros(((horizon-1)*self.obs_dim,)),
                ub=np.concatenate([np.ones(((horizon-1)*self.obs_dim,)) * 1e2]
                                  + [self.act_high]*horizon),
                lb=np.concatenate([-np.ones(((horizon-1)*self.obs_dim,)) * 1e2]
                                  + [self.act_low]*horizon),
            )
        elif self.method_str == 'ipopt_shooting':
            self.u_array_val = self._init_u_array()
            self.problem_obj = ShootingProblem(env=env, dynamics_model=dynamics_model, horizon=horizon)
            problem_config = dict(
                n=horizon*self.act_dim,
                m=0,
                problem_obj=self.problem_obj,
                lb=np.concatenate([self.act_low]*horizon),
                ub=np.concatenate([self.act_high]*horizon),
            )
        elif self.method_str == 'ipopt_shooting_w_policy':
            self.policy = policy
            # self.problem_obj = CollocationPolicyProblem(env=env, dynamics_model=dynamics_model, horizon=horizon)

        else:
            raise NotImplementedError

        self.nlp = nlp = ipopt.problem(**problem_config)
        nlp.addOption('mu_strategy', 'adaptive')
        nlp.addOption('tol', 1e-4)
        nlp.addOption('acceptable_tol', 1e-3)
        nlp.addOption('acceptable_iter', 5)
        nlp.addOption('max_iter', 30)
        # nlp.addOption('derivative_test', 'first-order')  # TEST FAILS

    @property
    def vectorized(self):
        return True

    def get_sinusoid_actions(self, action_space, t):
        actions = np.array([])
        for i in range(action_space):
            actions = np.append(actions, 0.5 * np.sin(i * t))
        return actions

    def _run_open_loop(self, u_array, init_obs):
        """

        :param u_array: np.array with shape (horizon, num_envs, act_dim)
        :param init_obs: np.array with shape (num_envs, obs_dim)
        :return: x_array: np.array with shape (horizon, num_envs, obs_dim)
        """
        returns = np.zeros((self.num_envs,))
        x_array = []
        obs = init_obs
        for i in range(self.horizon):
            x_array.append(obs)
            next_obs = self.dynamics_model.predict(obs=obs, act=u_array[i, :, :])
            reward = self.env.reward(obs=obs, acts=u_array[i, :, :], next_obs=next_obs)
            returns += self.discount**i * reward
            obs = next_obs

        x_array = np.stack(x_array, axis=0)
        return x_array, returns

    def _eval_open_loop(self, u_array, init_obs):
        if np.random.randn() < 0.01:
            return self.dubugger(u_array, init_obs)
        returns = np.zeros((self.num_envs,))
        _ = self.env_eval.reset(buffer={'observations': init_obs}, shuffle=False)
        for i in range(self.horizon):
            _, rewards, _, _ = self.env_eval.step(u_array[i, :, :])
            returns += rewards
        return returns

    def dubugger(self, u_array, init_obs):
        returns = np.zeros((self.num_envs,))
        _ = self.env_eval.reset(buffer={'observations': init_obs}, shuffle=False)
        for i in range(self.horizon):
            obs, rewards, _, _ = self.env_eval.step(u_array[i, :, :])
            logger.log(f"at {i}, obs = {obs}")
            returns += rewards
        return returns

    def get_actions(self, obs, verbose=True):
        """

        :param obs: (num_envs, obs_dim)
        :param verbose:
        :return:
        """
        if verbose:
            logger.logkv(f'ActNormBefore', np.square(self.u_array_val).sum())
        if self.method_str == 'ipopt_collocation':
            self.update_u_array_collocation(obs, verbose=verbose)
        elif self.method_str == 'ipopt_shooting':
            self.update_u_array_shooting(obs, verbose=verbose)

        optimized_actions = self.u_array_val
        self.shift_u_array(u_new=None)

        if verbose:
            logger.logkv(f'ActNormAfter', np.square(self.u_array_val).sum())
            returns = self._eval_open_loop(optimized_actions, obs)
            for env_idx in range(self.num_envs):
                logger.logkv(f'RetEnv-{env_idx}', returns[env_idx])
            logger.dumpkvs()

        return optimized_actions, []

    def update_u_array_shooting(self, obs, verbose=True):
        u_array = self.u_array_val
        for env_idx in range(self.num_envs):
            self.problem_obj.set_init_obs(obs[env_idx, :])
            inputs = self.problem_obj.get_inputs(u=u_array[:, env_idx, :])
            outputs, info = self.nlp.solve(inputs)
            outputs_u_array = self.problem_obj.get_u(outputs)
            u_clipped_pct = np.sum(np.abs(outputs_u_array) >= np.mean(self.act_high))/(self.horizon*self.act_dim)
            self.u_array_val[:, env_idx, :] = outputs_u_array
            if verbose and u_clipped_pct > 0:
                logger.logkv(f'u_clipped_pct-{env_idx}', u_clipped_pct)

    def update_u_array_collocation(self, obs, verbose=True):
        u_array = self.u_array_val
        x_array, returns = self._run_open_loop(u_array, obs)
        if hasattr(self.env, "get_goal_x_array"):
            x_array = self.env.get_goal_x_array(x_array)

        for env_idx in range(self.num_envs):
            # Feed in trajectory s[2:T], a[1:T], with s[1] == obs
            self.problem_obj.set_init_obs(obs[env_idx, :])
            inputs = self.problem_obj.get_inputs(x=x_array[1:, env_idx, :], u=u_array[:, env_idx, :])
            outputs, info = self.nlp.solve(inputs)
            outputs_x_array, outputs_u_array = self.problem_obj.get_x_u(outputs)
            outputs_u_array = np.clip(outputs_u_array, self.act_low, self.act_high)
            self.u_array_val[:, env_idx, :] = outputs_u_array

            u_clipped_pct = np.sum(np.abs(outputs_u_array) >= np.mean(self.act_high))/(self.horizon*self.act_dim)
            if verbose and u_clipped_pct > 0:
                logger.logkv(f'ReturnBefore-{env_idx}', returns[env_idx])
                logger.logkv(f'u_clipped_pct-{env_idx}', u_clipped_pct)

    def _sample_u(self):
        return np.clip(np.random.normal(size=(self.num_envs, self.act_dim), scale=0.1), a_min=self.act_low, a_max=self.act_high)
    
    def shift_u_array(self, u_new):
        if u_new is None:
            u_new = self._sample_u()
        self.u_array_val = np.concatenate([self.u_array_val[1:, :, :], u_new[None]])

    def _init_u_array(self):
        if self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.act_low, high=self.act_high,
                                             size=(self.horizon, self.num_envs, self.act_dim))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.4, size=(self.horizon, self.num_envs, self.act_dim))
            init_u_array = np.clip(init_u_array, a_min=self.act_low, a_max=self.act_high)
        else:
            raise NotImplementedError
        return init_u_array

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        """
        This funciton is called by sampler at the start of each trainer iteration.
        :param dones:
        :return:
        """
        self.u_array_val = self._init_u_array()

    def warm_reset(self, u_array):
        logger.log('planner resets with collected samples...')
        # if u_array is None or np.sum(np.abs(u_array) >= np.mean(self.act_high)) > 0.8 * (self.horizon*self.act_dim):
        #     u_array = self._init_u_array()
        # else:
        u_array = u_array[:self.horizon, :, :]
        self.u_array_val = u_array

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
