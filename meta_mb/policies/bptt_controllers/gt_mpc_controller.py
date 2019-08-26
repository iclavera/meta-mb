from meta_mb.utils.serializable import Serializable
import copy
import numpy as np


class GTMPCController(Serializable):
    def __init__(
            self,
            env,
            dynamics_model,
            num_rollouts=1,
            discount=1,
            method_str='cem',
            n_candidates=1024,
            horizon=10,
            num_cem_iters=8,
            percent_elites=0.1,
            alpha=0.25,
            num_particles=1,
            deterministic_policy=True,
    ):
        Serializable.quick_init(self, locals())

        self.dynamics_model = dynamics_model
        self.discount = discount
        self.method_str = method_str
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.num_cem_iters = num_cem_iters
        self.num_envs = num_rollouts
        self.num_elites = max(int(percent_elites * n_candidates), 1)
        self.alpha = alpha
        self.num_particles = num_particles
        self.deterministic_policy = deterministic_policy

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        self._env = copy.deepcopy(env)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation, verbose=True):
        actions, _ = self.get_actions(observation[None], verbose=verbose)
        return actions[0], []

    def get_actions(self, observations, verbose=True):
        if self.method_str == "cem":
            return self._get_actions_cem(observations, verbose)
        if self.method_str == "rs":
            return self._get_actions_rs(observations, verbose)

    def _get_actions_cem(self, observations, verbose):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        horizon = self.horizon
        n_candidates = self.n_candidates
        num_envs = self.num_envs

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

            for i in range(num_envs*n_candidates):
                _ = self._env.reset_from_obs(observations[i // n_candidates, :])
                for t in range(horizon):
                    _, reward, _, _ = self._env.step(act[t, i, :])
                    returns[i] += self.discount**t * reward

            # Re-fit belief to the best ones
            returns = np.reshape(returns, (num_envs, n_candidates))
            act = np.reshape(act, (horizon, num_envs, n_candidates, act_dim))
            act = np.transpose(act, (1, 2, 0, 3))  # (num_envs, n_candidates, horizon, act_dim)
            indices = np.argsort(-returns, axis=1)[:, :self.num_elites]  # (num_envs, num_elites)
            elite_actions = np.stack([act[env_idx, indices[env_idx]] for env_idx in range(self.num_envs)], axis=0)
            elite_actions = np.transpose(elite_actions, (2, 0, 1, 3))  # (horizon, num_envs, n_candidates, act_dim)
            elite_mean = np.mean(elite_actions, axis=2, keepdims=True)
            elite_var = np.var(elite_actions, axis=2, keepdims=True)
            mean = mean * self.alpha + elite_mean * (1 - self.alpha)
            var = var * self.alpha + elite_var * (1 - self.alpha)

        optimized_actions_mean = mean[0, :, 0, :]
        if self.deterministic_policy:
            optimized_actions = optimized_actions_mean
        else:
            optimized_actions_var = var[0, :, 0, :]
            optimized_actions = mean + np.random.normal(size=np.shape(optimized_actions_mean)) * np.sqrt(optimized_actions_var)

        return optimized_actions, []

    def _get_actions_rs(self, observations, verbose):
        num_envs = self.num_envs
        n_candidates = self.n_candidates
        horizon = self.horizon

        returns = np.zeros((num_envs*n_candidates,))
        act = np.random.uniform(
            low=self.act_low,
            high=self.act_high,
            size=((horizon, num_envs*n_candidates, self.act_dim))
        )

        for i in range(num_envs*n_candidates):
            _ = self._env.reset_from_obs(observations[i // n_candidates, :])
            for t in range(horizon):
                _, reward, _, _ = self._env.step(act[t, i, :])
                returns[i] += self.discount**t * reward

        returns = np.reshape(returns, (num_envs, n_candidates))
        cand_a = np.reshape(act[0], newshape=(num_envs, n_candidates, self.act_dim))
        return cand_a[range(self.num_envs), np.argmax(returns, axis=1)], []

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
