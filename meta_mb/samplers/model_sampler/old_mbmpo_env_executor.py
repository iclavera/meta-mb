import numpy as np
import pickle as pickle
from meta_mb.utils import stack_tensor_dict_list


class ModelVecEnvExecutor(object):
    def __init__(self, env, model, max_path_length, n_parallel):
        self.env = env
        self.model = model

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        # check whether env has done function
        self.has_done_fn = hasattr(self.unwrapped_env, 'done')

        self.n_parallel = n_parallel
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self.ts = np.zeros(n_parallel, dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_n):
        # use the model to make (predicted) steps
        prev_obs = self.current_obs
        next_obs = self.model.predict(prev_obs, action_n)

        rewards = self.unwrapped_env.reward(prev_obs, action_n, next_obs)

        if self.has_done_fn:
            dones = self.unwrapped_env.done(next_obs)
        else:
            dones = np.asarray([False for _ in range(self.n_parallel)])

        env_infos = [{} for _ in range(action_n.shape[0])]

        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                next_obs[i] = self.env.reset()
                self.ts[i] = 0

        self.current_obs = next_obs

        # transform obs to lists
        next_obs = [np.squeeze(o) for o in np.vsplit(next_obs, next_obs.shape[0])]
        return next_obs, list(rewards), list(dones), stack_tensor_dict_list(env_infos) #lists

    def reset(self):
        results = [self.env.reset() for _ in range(self.n_parallel)] # get initial observation from environment
        self.current_obs = np.stack(results, axis=0)
        assert self.current_obs.ndim == 2
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return self.n_parallel

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
