from meta_mb.utils.serializable import Serializable
import numpy as np
from meta_mb.logger import logger


class DyniLQRController(Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            planner,
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
        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"
        self.planner = planner
        self.planner.reset_u_array(self._init_u_array())

    @property
    def vectorized(self):
        return True

    def get_sinusoid_actions(self, action_space, t):
        actions = np.array([])
        for i in range(action_space):
            actions = np.append(actions, 0.5 * np.sin(i * t))
        return actions

    def get_actions(self, obs):
        return self.planner.get_actions(obs)

    def _init_u_array(self):
        if self.initializer_str == 'cem':
            init_u_array = self._init_u_array_cem()
        elif self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.act_low, high=self.act_high,
                                             size=(self.horizon, self.num_envs, self.action_space_dims))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.num_envs, self.action_space_dims))
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
        mean = np.ones(shape=(self.horizon, self.num_envs, 1, self.action_space_dims)) \
               * (self.act_high + self.act_low) / 2
        std = np.ones(shape=(self.horizon, self.num_envs, 1, self.action_space_dims)) \
              * (self.act_high - self.act_low) / 4

        for itr in range(10):
            lb_dist, ub_dist = mean - self.act_low, self.act_high - mean
            # constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)
            # std = np.sqrt(constrained_var)
            std = np.minimum(lb_dist/2, ub_dist/2, std)
            act = mean + np.random.normal(size=(self.horizon, self.num_envs, self.n_candidates, self.action_space_dims)) * std
            act = np.clip(act, self.act_low, self.act_high)
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

    def warm_reset(self, u_array):
        logger.log('planner resets with collected samples...')
        self.planner.reset_u_array(u_array[:self.horizon, :, :])

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
