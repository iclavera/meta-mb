from meta_mb.utils.serializable import Serializable
import numpy as np
from meta_mb.logger import logger
from meta_mb.policies.ipopt_problems.ipopt_dyn_collocation_problem import CollocationProblem
import ipopt
import tensorflow as tf


class DynCollocationController(Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            num_rollouts,
            discount,
            n_parallel,
            initializer_str,
            n_candidates,
            horizon,
            percent_elites=0.1,
            alpha=0.25,
            verbose=True,
    ):
        Serializable.quick_init(self, locals())
        self.env = env
        self.dynamics_model = dynamics_model
        self.discount = discount
        self.initializer_str = initializer_str
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
        self.nlp = nlp = ipopt.problem(**problem_config)
        nlp.addOption('mu_strategy', 'adaptive')
        nlp.addOption('tol', 1e-5)
        nlp.addOption('max_iter', 100)
        # nlp.addOption('derivative_test', 'first-order')  # SLOW

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

    def get_actions(self, obs, verbose=True):
        """

        :param obs: (num_envs, obs_dim)
        :param verbose:
        :return: (num_envs, act_dim)
        """
        u_array = self.u_array_val
        x_array, returns = self._run_open_loop(u_array, obs)
        logger.log('PrevReturn', returns)

        for env_idx in range(self.num_envs):
            # Feed in trajectory s[2:T], a[1:T], with s[1] == obs
            self.problem_obj.set_init_obs(obs[env_idx, :])
            inputs = self.problem_obj.get_inputs(x=x_array[1:, env_idx, :], u=u_array[:, env_idx, :])
            outputs, info = self.nlp.solve(inputs)
            outputs_x_array, outputs_u_array = self.problem_obj.get_x_u(outputs)
            self.u_array_val[:, env_idx, :] = outputs_u_array

            # logging
            if verbose:
                logger.logkv('Action100%', np.max(outputs_u_array, axis=None))
                logger.logkv('Action0%', np.min(outputs_u_array, axis=None))
                logger.logkv('Action75%', np.percentile(outputs_u_array, q=75, axis=None))
                logger.logkv('Action25%', np.percentile(outputs_u_array, q=25, axis=None))
                logger.logkv('Obs100%', np.max(outputs_x_array, axis=None))
                logger.logkv('Obs0%', np.min(outputs_x_array, axis=None))

        optimized_action = self.u_array_val[0, :, :]
        # shift
        self.shift_u_array(u_new=None)

        # neg_return, reg_loss = self._compute_collocation_loss(np.concatenate([[obs], self.running_s_array]),
        #                                                        self.running_a_array)
        # logger.logkv('ColNegReturn', neg_return)
        # logger.logkv('ColRegLoss', reg_loss)

        return optimized_action, []
    
    def _sample_u(self):
        return np.clip(np.random.normal(size=(self.num_envs, self.act_dim), scale=0.1), a_min=self.act_low, a_max=self.act_high)
    
    def shift_u_array(self, u_new):
        if u_new is None:
            u_new = self._sample_u()
        self.u_array_val = np.concatenate([self.u_array_val[1:, :, :], u_new[None]])

    def _init_u_array(self):
        if self.initializer_str == 'cem':
            init_u_array = self._init_u_array_cem()
        elif self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.act_low, high=self.act_high,
                                             size=(self.horizon, self.num_envs, self.act_dim))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.num_envs, self.act_dim))
            init_u_array = np.clip(init_u_array, a_min=self.act_low, a_max=self.act_high)
        else:
            raise NotImplementedError
        return init_u_array

    def _init_u_array_cem(self):

        """
        This function assumes ground truth.
        :return:
        """
        raise NotImplementedError
        assert self.num_envs == 1
        # _env = IterativeEnvExecutor(self._env, self.num_envs*self.n_candidates, self.max_path_length)
        mean = np.ones(shape=(self.horizon, self.num_envs, 1, self.act_dim)) \
               * (self.act_high + self.act_low) / 2
        std = np.ones(shape=(self.horizon, self.num_envs, 1, self.act_dim)) \
              * (self.act_high - self.act_low) / 4

        for itr in range(10):
            lb_dist, ub_dist = mean - self.act_low, self.act_high - mean
            # constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            # std = np.sqrt(constrained_var)
            std = np.minimum(lb_dist/2, ub_dist/2, std)
            act = mean + np.random.normal(size=(self.horizon, self.num_envs, self.n_candidates, self.act_dim)) * std
            act = np.clip(act, self.act_low, self.act_high)
            act = np.reshape(act, (self.horizon, self.num_envs*self.n_candidates, self.act_dim))

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
            act = np.reshape(act, (self.horizon, self.num_envs, self.n_candidates, self.act_dim))
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

    def warm_reset(self, u_array):
        logger.log('planner resets with collected samples...')
        if u_array is None or np.sum(np.abs(u_array) >= np.mean(self.act_high)) > 0.8 * (self.horizon*self.act_dim):
            u_array = self._init_u_array()
        else:
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
