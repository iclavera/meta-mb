import numpy as np

import ray
import pickle


@ray.remote
class RemoteEnv(object):
    def __init__(self, env_pickled, num_rollouts, max_path_length, seed):
        self._envs = [pickle.loads(env_pickled) for _ in range(num_rollouts)]
        self._num_envs = num_rollouts
        self._max_path_length = max_path_length
        self._seed = seed
        self.ts = np.zeros(num_rollouts, dtype='int')

    def step(self, act_na):
        """

        :param act_na: (np.array)
        :return: (tuple): a length 4 tuple of lists, each list has np.array or scalar elements
        """
        all_results = [env.step(a) for a, env in zip(act_na, self._envs)]
        obs, rewards, dones, infos = map(list, zip(*all_results))

        if isinstance(obs[0], dict):
            obs = list(map(lambda obs_dict: obs_dict['observation'], obs))

        # reset env when done or max_path_length reached
        self.ts += 1
        dones = np.logical_or(self.ts >= self._max_path_length, dones).tolist()

        for i in np.argwhere(dones).flatten():
            obs[i] = self._envs[i].reset(self._envs[i].goal)
            if isinstance(obs[i], dict):
                obs[i] = obs[i]['observation']
            self.ts[i] = 0

        return obs, rewards, dones, infos

    def reset(self, goal_ng):
        """

        :param goal_ng: (np.array)
        :return: (list)
        """
        init_ob_no = [env.reset(goal) for goal, env in zip(goal_ng, self._envs)]
        if isinstance(init_ob_no[0], dict):
            init_ob_no = list(map(lambda obs_dict: obs_dict['observation'], init_ob_no))
        self.ts[:] = 0
        return init_ob_no