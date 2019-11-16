import tensorflow as tf
import pickle


class RemoteEnv(object):
    def __init__(self, env_pickled, num_rollouts, max_path_length, seed):
        self._envs = [pickle.loads(env_pickled) for _ in range(num_rollouts)]
        self._num_rollouts = num_rollouts
        self._max_path_length = max_path_length
        self._seed = seed

    def step(self, act_na):
        all_results = [env.step(a) for a, env in zip(act_na, self._envs)]
        obs, rewards, dones, infos = list(zip(*all_results))

        if isinstance(obs[0], dict):
            obs = list(map(lambda obs_dict: obs_dict['observation'], obs))

        # reset env when done or max_path_length reached
        self.ts += 1
        for i in range(n_envs):
            if dones[i] or (ts[i] >= max_path_length):
                dones[i] = True
                obs[i] = envs[i].reset()
                ts[i] = 0

